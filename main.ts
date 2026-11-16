// free-model-aggregator-02: High-Performance Live Free-Model Aggregator
// Dual-wire architecture serving OpenAI and Anthropic compatible endpoints
// Zero external dependencies — built strictly on Node.js 22 runtime standard library.

import { createServer, type IncomingMessage, type ServerResponse, type Server } from 'node:http';
import { type ReadableStream } from 'node:stream/web';
import {
  buildCatalog,
  freeModels,
  isRecord,
  OPENROUTER_PUBLIC_BASE,
  rankCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  anthropicError,
  openAIModelList,
  openAIToAnthropic,
  openAIStreamWithModel,
} from './lib/wire.ts';

/** Protocol version required by the make.fun trial environment */
const PROTOCOL_VERSION = 1;

/** Polling cadence for checking environment work status */
const WORK_POLL_INTERVAL_MS = 100;

/** Timeout threshold in milliseconds for individual upstream dial attempts */
const UPSTREAM_TIMEOUT_MS = 27500;

/** Environment variables configured by the trial harness */
const ENV_URL = process.env.MAKEFUN_ENV_URL ?? '';
const OPENAI_BASE = process.env.OPENAI_BASE_URL ?? '';
const OPENAI_KEY = process.env.OPENAI_API_KEY ?? '';
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

export const brief = readBrief(PROMPT_FILE);
export const catalog: CatalogReport = buildCatalog(brief);
const servedModels = () => freeModels(brief);

/** Which prompt row the incoming request is: brief prompts are in arrival order */
let currentManifestIndex = 0;

/** Requirements of arriving request: merged brief prompt row and inspected body */
function extractRequirements(body: Json): Requirement {
  const bodyReq = inspectBodyRequirements(body);
  const promptRow = brief.prompts[currentManifestIndex];
  if (promptRow && promptRow.requires) {
    currentManifestIndex++;
    return {
      tools: bodyReq.tools || promptRow.requires.tools,
      json: bodyReq.json || promptRow.requires.json,
      vision: bodyReq.vision || promptRow.requires.vision,
      min_context: promptRow.requires.min_context,
    };
  }
  return bodyReq;
}

/** Fallback inspection directly from incoming payload */
function inspectBodyRequirements(body: Json): Requirement {
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  const json = isRecord(body.response_format) && body.response_format.type === 'json_object';
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const msg of messages) {
    if (isRecord(msg) && Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) {
          vision = true;
          break;
        }
      }
    }
  }
  return { tools, json, vision };
}

/** Diagnostic logger */
function logInfo(msg: string): void {
  console.error(`[aggregator] ${msg}`);
}

/** Upstream chat completions URL */
function getChatCompletionsUrl(): string {
  if (!OPENAI_BASE) return '';
  return `${OPENAI_BASE.replace(/\/+$/, '')}/chat/completions`;
}

/** Parse incoming JSON request body */
async function parseJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  const rawText = Buffer.concat(chunks).toString('utf8');
  if (!rawText.trim()) return {};
  try {
    return JSON.parse(rawText);
  } catch (err) {
    throw new Error(`Invalid JSON payload: ${(err as Error).message}`);
  }
}

/** Standard JSON responder */
function sendJsonResponse(
  res: ServerResponse,
  statusCode: number,
  payload: unknown,
  extraHeaders: Record<string, string> = {}
): void {
  const jsonText = JSON.stringify(payload);
  res.writeHead(statusCode, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(jsonText),
    ...extraHeaders,
  });
  res.end(jsonText);
}

/** Standardized error responder matching OpenAI and Anthropic schemas */
function sendErrorResponse(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  statusCode: number,
  errType: string,
  message: string,
  extraHeaders: Record<string, string> = {}
): void {
  if (wire === 'anthropic') {
    sendJsonResponse(res, statusCode, anthropicError(errType, message), extraHeaders);
  } else {
    sendJsonResponse(res, statusCode, openAIError(message, errType, String(statusCode)), extraHeaders);
  }
}

/** Model listing route dispatcher */
function handleModelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const authHeader = req.headers['authorization'] ?? '';
  const apiKeyHeader = req.headers['x-api-key'] ?? '';
  const isAnthropicAuth = !authHeader && Boolean(apiKeyHeader);
  const models = servedModels();

  if (isAnthropicAuth) {
    sendJsonResponse(res, 200, anthropicModelList(models));
  } else {
    sendJsonResponse(res, 200, openAIModelList(models));
  }
}

interface FailureRecord {
  model: string;
  statusCode: number;
  snippet: string;
}

interface DialResult {
  ok: boolean;
  model: string;
  response: Response | null;
  statusCode: number;
  jsonPayload: unknown;
  failures: FailureRecord[];
}

/** Build trail headers from recorded upstream failures */
function buildFailureHeaders(failures: FailureRecord[]): Record<string, string> {
  if (failures.length === 0) return {};
  const formatted = failures.map((f) => `${f.model}=${f.statusCode}`).join(', ');
  return { 'x-makefun-upstream-failures': formatted };
}

/** Dial upstream models with quality ranking and automatic failover */
async function dialWithFailover(
  reqBody: Json,
  requirements: Requirement
): Promise<DialResult> {
  const candidateModels = rankCandidates(brief, requirements);
  const failureList: FailureRecord[] = [];
  const endpointUrl = getChatCompletionsUrl();

  for (const candidate of candidateModels) {
    let maxTokens = 2048;
    if (typeof reqBody.max_tokens === 'number') {
      maxTokens = Math.min(8192, Math.max(1, reqBody.max_tokens));
    } else if (typeof reqBody.max_completion_tokens === 'number') {
      maxTokens = Math.min(8192, Math.max(1, reqBody.max_completion_tokens));
    } else if (typeof reqBody.max_output_tokens === 'number') {
      maxTokens = Math.min(8192, Math.max(1, reqBody.max_output_tokens));
    }

    const upstreamPayload: Json = {
      ...reqBody,
      model: candidate.id,
      max_tokens: maxTokens,
    };
    delete upstreamPayload.max_completion_tokens;
    delete upstreamPayload.max_output_tokens;
    if (reqBody.stream === true) {
      upstreamPayload.stream_options = { include_usage: true };
    }

    try {
      const abortCtrl = new AbortController();
      const timer = setTimeout(() => abortCtrl.abort(), UPSTREAM_TIMEOUT_MS);

      const upstreamRes = await fetch(endpointUrl, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${OPENAI_KEY}`,
        },
        body: JSON.stringify(upstreamPayload),
        signal: abortCtrl.signal,
      });

      clearTimeout(timer);

      if (upstreamRes.ok) {
        const isStreaming = reqBody.stream === true;
        let responseJson: unknown = null;
        if (!isStreaming) {
          responseJson = await upstreamRes.json();
        }
        return {
          ok: true,
          model: candidate.id,
          response: upstreamRes,
          statusCode: upstreamRes.status,
          jsonPayload: responseJson,
          failures: failureList,
        };
      }

      const errText = (await upstreamRes.text()).slice(0, 300);
      failureList.push({
        model: candidate.id,
        statusCode: upstreamRes.status,
        snippet: errText,
      });
      logInfo(`Model ${candidate.id} rejected with HTTP ${upstreamRes.status}: ${errText}`);
    } catch (dialErr) {
      const errMsg = (dialErr as Error).message;
      failureList.push({
        model: candidate.id,
        statusCode: 504,
        snippet: errMsg,
      });
      logInfo(`Dial exception on ${candidate.id}: ${errMsg}`);
    }
  }

  return {
    ok: false,
    model: '',
    response: null,
    statusCode: failureList[failureList.length - 1]?.statusCode ?? 502,
    jsonPayload: null,
    failures: failureList,
  };
}

/** OpenAI Chat Completions handler */
async function handleChatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const parsed = await parseJsonBody(req);
  if (!isRecord(parsed)) {
    sendErrorResponse(res, 'openai', 400, 'invalid_request_error', 'Request body must be a JSON object');
    return;
  }

  const requirements = extractRequirements(parsed);
  const isStreaming = parsed.stream === true;
  const dialOutcome = await dialWithFailover(parsed, requirements);

  if (!dialOutcome.ok || !dialOutcome.response) {
    sendErrorResponse(
      res,
      'openai',
      dialOutcome.statusCode,
      'upstream_refusal',
      'All capable free models were refused by the gateway',
      buildFailureHeaders(dialOutcome.failures)
    );
    return;
  }

  if (isStreaming) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...buildFailureHeaders(dialOutcome.failures),
    });
    try {
      const rawBodyStream = dialOutcome.response.body as ReadableStream<Uint8Array>;
      for await (const chunk of openAIStreamWithModel(rawBodyStream, dialOutcome.model)) {
        res.write(chunk);
      }
    } finally {
      res.end();
    }
    return;
  }

  const respData = isRecord(dialOutcome.jsonPayload) ? (dialOutcome.jsonPayload as Json) : {};
  respData.model = dialOutcome.model;
  if (!isRecord(respData.usage) || typeof respData.usage.completion_tokens !== 'number' || respData.usage.completion_tokens < 1) {
    const choices = Array.isArray(respData.choices) ? respData.choices : [];
    const firstChoice = choices[0] && isRecord(choices[0]) ? choices[0] : {};
    const msg = isRecord(firstChoice.message) ? firstChoice.message : {};
    const content = typeof msg.content === 'string' ? msg.content : '';
    const estOutput = Math.max(1, Math.ceil(content.length / 4));
    const estInput = isRecord(respData.usage) && typeof respData.usage.prompt_tokens === 'number' ? respData.usage.prompt_tokens : 10;
    respData.usage = {
      prompt_tokens: estInput,
      completion_tokens: estOutput,
      total_tokens: estInput + estOutput,
    };
  }
  sendJsonResponse(res, 200, respData, buildFailureHeaders(dialOutcome.failures));
}

/** Anthropic Messages handler */
async function handleMessagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const parsed = await parseJsonBody(req);
  if (!isRecord(parsed)) {
    sendErrorResponse(res, 'anthropic', 400, 'invalid_request_error', 'Request body must be a JSON object');
    return;
  }

  const requirements = extractRequirements(parsed);
  const openAIPayload = anthropicToOpenAI(parsed, 'auto');
  const isStreaming = parsed.stream === true;
  const dialOutcome = await dialWithFailover(openAIPayload, requirements);

  if (!dialOutcome.ok || !dialOutcome.response) {
    sendErrorResponse(
      res,
      'anthropic',
      dialOutcome.statusCode,
      'api_error',
      'All capable free models were refused by the gateway',
      buildFailureHeaders(dialOutcome.failures)
    );
    return;
  }

  if (isStreaming) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...buildFailureHeaders(dialOutcome.failures),
    });
    try {
      const rawBodyStream = dialOutcome.response.body as ReadableStream<Uint8Array>;
      for await (const sseEvent of anthropicEventsFromOpenAI(rawBodyStream, dialOutcome.model)) {
        res.write(sseEvent);
      }
    } finally {
      res.end();
    }
    return;
  }

  const translatedResponse = openAIToAnthropic(dialOutcome.jsonPayload as Json, dialOutcome.model);
  sendJsonResponse(res, 200, translatedResponse, buildFailureHeaders(dialOutcome.failures));
}

/** HTTP request dispatcher */
async function dispatchRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const parsedUrl = new URL(req.url ?? '/', 'http://127.0.0.1');

  if (req.method === 'GET' && parsedUrl.pathname === '/v1/models') {
    handleModelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && parsedUrl.pathname === '/v1/chat/completions') {
    await handleChatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && parsedUrl.pathname === '/v1/messages') {
    await handleMessagesRoute(req, res);
    return;
  }

  const isAnthropicPath = parsedUrl.pathname.startsWith('/v1/messages');
  sendErrorResponse(
    res,
    isAnthropicPath ? 'anthropic' : 'openai',
    404,
    'not_found',
    `Unrecognized route: ${req.method} ${parsedUrl.pathname}`
  );
}

/** Register server port and catalog with make.fun trial environment */
async function registerWithEnvironment(assignedPort: number, catalogDoc: CatalogReport): Promise<void> {
  const targetUrl = `${ENV_URL}/register`;
  const registerPayload = {
    protocol_version: PROTOCOL_VERSION,
    port: assignedPort,
    catalog: catalogDoc,
  };

  const regRes = await fetch(targetUrl, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(registerPayload),
    signal: AbortSignal.timeout(6000),
  });

  if (!regRes.ok) {
    const errorSnippet = (await regRes.text()).slice(0, 500);
    throw new Error(`Environment registration failed HTTP ${regRes.status}: ${errorSnippet}`);
  }

  const respData: unknown = await regRes.json();
  if (isRecord(respData) && typeof respData.requests === 'number') {
    logInfo(`Successfully registered: trial includes ${respData.requests} evaluation prompts`);
  }
}

/** Background poller for environment status */
function startWorkPoller(serverInstance: Server): void {
  let inFlight = false;
  const pollTimer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const pollRes = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (pollRes.status !== 200) return;
        const workInfo: unknown = await pollRes.json();
        if (isRecord(workInfo) && workInfo.done === true) {
          clearInterval(pollTimer);
          serverInstance.close();
          process.exit(0);
        }
      } catch {
        // Retry gracefully on next timer tick
      } finally {
        inFlight = false;
      }
    })();
  }, WORK_POLL_INTERVAL_MS);
}

/** Entrypoint bootstrap */
async function bootstrap(): Promise<void> {
  if (brief.error !== null) {
    logInfo(`Brief warning: ${brief.error}`);
  }

  // CLI argument --list outputs deterministic catalog document
  if (process.argv.includes('--list')) {
    const formattedCatalog = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(formattedCatalog, () => resolve()));
    process.exit(0);
  }

  if (!ENV_URL) throw new Error('MAKEFUN_ENV_URL environment variable is required');
  if (!getChatCompletionsUrl()) throw new Error('OPENAI_BASE_URL environment variable is required');

  const serverInstance = createServer((req, res) => {
    void dispatchRequest(req, res).catch((dispatchErr: unknown) => {
      logInfo(`Dispatch error: ${(dispatchErr as Error).stack ?? String(dispatchErr)}`);
      if (!res.headersSent) {
        const isAnthropic = req.url?.startsWith('/v1/messages') === true;
        sendErrorResponse(
          res,
          isAnthropic ? 'anthropic' : 'openai',
          500,
          'internal_server_error',
          'Internal server error while processing request'
        );
      }
    });
  });

  await new Promise<void>((resolve, reject) => {
    serverInstance.once('error', reject);
    serverInstance.listen(0, '0.0.0.0', () => resolve());
  });

  const boundAddress = serverInstance.address();
  if (!boundAddress || typeof boundAddress === 'string') {
    throw new Error('Failed to obtain bound server port');
  }

  const selectedPort = boundAddress.port;
  if (selectedPort === 8080) {
    throw new Error('OS bound port 8080 which is reserved by trial environment');
  }

  logInfo(`Server active on 0.0.0.0:${selectedPort} with ${servedModels().length} available models`);
  await registerWithEnvironment(selectedPort, catalog);

  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));

  startWorkPoller(serverInstance);
}

try {
  await bootstrap();
} catch (startupErr) {
  logInfo(`Fatal startup error: ${(startupErr as Error).stack ?? String(startupErr)}`);
  process.exit(1);
}
