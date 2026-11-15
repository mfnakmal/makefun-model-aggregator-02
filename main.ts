// free-model-aggregator-02: one server, two client wires, and a real free model behind
// every answer.
//
// The trial runs this server as `node main.ts` from the repository root, with Node 22's
// standard library alone (nothing is installed). The env is at $MAKEFUN_ENV_URL; model
// calls go through the grader's gateway at $OPENAI_BASE_URL, an OpenAI-compatible
// endpoint that reaches OpenRouter on the key pushed to the competition. The brief at
// $MAKEFUN_PROMPT_FILE carries the free model list (each model's gateway id, its
// rating, what it can do) and the six prompts' manifest.
//
// The server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves GET /v1/models (answering in the shape of the auth header it got), POST
//      /v1/chat/completions and POST /v1/messages, each with a JSON body or with
//      server-sent events;
//   4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY
//      in the sandbox is a dummy the gateway swaps for the entrant's own OpenRouter key;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says
//      {"done": true}. A process still running when the wall ends scores zero.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it
//   actually dialled. The house records every gateway call and scores the model that
//   really answered. This server sends the dialled id in the gateway request and puts
//   that same id back on every answer and every streamed chunk (the provider's own
//   answer spells it without the gateway's `openrouter/` prefix), so its claim is the
//   call it made, never another.
//
// ROUTING AND FAILOVER (where the score is won)
//   - quality (60%): candidates are the brief's free models that meet the request's
//     requirements, ranked by the brief's rating_milli, highest first, so the first
//     model tried is the best one for that prompt. A JSON prompt goes to the best
//     JSON-capable model, a tools prompt to the best tools-capable one, the rest to the
//     best overall. Requirements are read off the request body itself, so the routing
//     never depends on the manifest's order.
//   - failover (10%): a 429 or 5xx is retried on the next-best capable model, never by
//     giving up while a capable model is untried; every refusal is reported on the
//     response header x-makefun-upstream-failures, the only way the failover column
//     sees a retried-then-served refusal. Within one prompt each model is tried at most
//     once, so a refusing model is never hammered; across prompts the ranking is fresh,
//     so a rate limit that clears lets the best model serve again. A 4xx other than 429
//     is not retried: it would fail the same way on every model.
//   - the catalog (5%): the register body and both GET /v1/models answers list exactly
//     the fourteen free ids, once each, no more and no fewer.
//   - determinism: every choice is a function of the brief, the requests and the
//     gateway answers seen so far. No wall clock (fixed timestamps in the listings, no
//     Date.now) and no unseeded randomness; the only sleeps are fixed backoffs between
//     failover attempts, which change no decision.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  rankCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The gateway's per-request output cap: a request naming more, or none, is refused. */
const MAX_OUTPUT_TOKENS = 8192;
/** The output cap named when a request carries none. */
const DEFAULT_OUTPUT_TOKENS = 1024;
/** How long one gateway call may take before it is abandoned for the next model. */
const GATEWAY_TIMEOUT_MS = 20_000;
/** The pause between one model's refusal and the next model's attempt. */
const FAILOVER_BACKOFF_MS = 250;
/** The most gateway calls one prompt may spend: six prompts stay under the trial's 60-call cap. */
const MAX_ATTEMPTS_PER_PROMPT = 5;
/** The most refusal rows the trail header carries, as the brief allows. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root
 * (`…/v1` on the trial network); a base that does not already end in `/v1` gets one, so
 * both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of one request, read off its own body: what its model must support. */
function requirementsFromBody(body: Json): Requirement {
  const requires: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') requires.json = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'image' || part.type === 'image_url') requires.vision = true;
    }
  }
  return requires;
}

/**
 * A Chat Completions body with a legal output cap: the gateway refuses a request with
 * no cap, or one over its per-request cap, so a missing cap becomes the default and an
 * over-large one is clamped. The caller's other fields pass through unchanged.
 */
function withOutputCap(body: Json): Json {
  const out: Json = { ...body };
  const tokens = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : null;
  if (tokens === null || !Number.isFinite(tokens) || tokens < 1) {
    out.max_tokens = DEFAULT_OUTPUT_TOKENS;
    delete out.max_completion_tokens;
  } else if (tokens > MAX_OUTPUT_TOKENS) {
    if (typeof out.max_tokens === 'number') out.max_tokens = MAX_OUTPUT_TOKENS;
    else out.max_completion_tokens = MAX_OUTPUT_TOKENS;
  }
  return out;
}

/** One refusal, as the trail header reports it. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header for a prompt's refusals, or nothing when there were none. */
function trailHeaders(failures: readonly Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS)) };
}

/** A refused gateway call's status and short code, from its error body when it has one. */
async function refusalOf(res: Response): Promise<{ status: number; code: string }> {
  const status = res.status;
  let code = `http_${status}`;
  try {
    const text = await res.text();
    if (text !== '') {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error)) {
        if (typeof parsed.error.code === 'string' && parsed.error.code !== '') code = parsed.error.code;
        else if (typeof parsed.error.type === 'string' && parsed.error.type !== '') code = parsed.error.type;
      }
    }
  } catch {
    // Not JSON, or unreadable: the HTTP status is the code.
  }
  return { status, code };
}

/** POST one request to the gateway's Chat Completions route, naming the model to dial. */
async function dial(model: string, body: Json, stream: boolean): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (stream) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
  });
}

type UpstreamResult =
  | { served: true; model: string; failures: Failure[]; response: Response; json: Json | null }
  | { served: false; failures: Failure[]; status: number; code: string };

/**
 * Dial the gateway for one prompt, failing over through the ranked candidates.
 *
 * The first candidate is the best capable model; a 429 or 5xx moves to the next-best
 * capable model, a network error or a broken answer does the same, and any other 4xx
 * stops the attempt (it would fail the same way on every model). At most five calls
 * are spent on one prompt, so six prompts stay under the trial's 60-call cap. Every
 * refusal is collected for the trail header. A streamed answer is returned as the live
 * response (its body is relayed, never buffered); a JSON answer is parsed and validated
 * before it counts as served, so a broken 200 never stands as the prompt's answer.
 */
async function callUpstream(candidates: ModelRow[], buildBody: (model: string) => Json, stream: boolean): Promise<UpstreamResult> {
  const failures: Failure[] = [];
  const attempts = Math.min(MAX_ATTEMPTS_PER_PROMPT, candidates.length);
  for (let i = 0; i < attempts; i++) {
    const model = candidates[i];
    let res: Response;
    try {
      res = await dial(model.id, buildBody(model.id), stream);
    } catch {
      failures.push({ status: 0, code: 'network_error', model: model.id });
      await sleep(FAILOVER_BACKOFF_MS);
      continue;
    }
    if (!res.ok) {
      const { status, code } = await refusalOf(res);
      failures.push({ status, code, model: model.id });
      if (status !== 429 && status < 500) break;
      await sleep(FAILOVER_BACKOFF_MS);
      continue;
    }
    if (stream) return { served: true, model: model.id, failures, response: res, json: null };
    let text: string;
    try {
      text = await res.text();
    } catch {
      failures.push({ status: 200, code: 'bad_body', model: model.id });
      await sleep(FAILOVER_BACKOFF_MS);
      continue;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      failures.push({ status: 200, code: 'bad_json', model: model.id });
      await sleep(FAILOVER_BACKOFF_MS);
      continue;
    }
    if (!isRecord(parsed)) {
      failures.push({ status: 200, code: 'bad_json', model: model.id });
      await sleep(FAILOVER_BACKOFF_MS);
      continue;
    }
    return { served: true, model: model.id, failures, response: res, json: parsed };
  }
  const last = failures[failures.length - 1];
  return { served: false, failures, status: last !== undefined ? last.status : 503, code: last !== undefined ? last.code : 'failover_exhausted' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type, null), extraHeaders);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankCandidates(brief, requirementsFromBody(body));
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const stream = body.stream === true;
  const upstreamBody = withOutputCap(body);
  if (stream) upstreamBody.stream_options = { include_usage: true };
  const result = await callUpstream(candidates, (model) => ({ ...upstreamBody, model }), stream);
  if (!result.served) {
    log(`chat: every candidate refused (${result.code}); the prompt is lost`);
    return sendError(res, 'openai', 502, 'upstream_error', `the gateway refused every capable model (${result.code})`, trailHeaders(result.failures));
  }
  if (stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeaders(result.failures),
    });
    try {
      for await (const frame of openAIStreamWithModel(result.response.body as ReadableStream<Uint8Array>, result.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(result.json as Json, result.model), trailHeaders(result.failures));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = rankCandidates(brief, requirementsFromBody(body));
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const stream = body.stream === true;
  const result = await callUpstream(candidates, (model) => withOutputCap(anthropicToOpenAI(body, model)), stream);
  if (!result.served) {
    log(`messages: every candidate refused (${result.code}); the prompt is lost`);
    return sendError(res, 'anthropic', 502, 'api_error', `the gateway refused every capable model (${result.code})`, trailHeaders(result.failures));
  }
  if (stream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeaders(result.failures),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(result.response.body as ReadableStream<Uint8Array>, result.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(result.json as Json, result.model), trailHeaders(result.failures));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built
    // object and stop before any server starts. A process whose stdout is not flushed
    // loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
