// free-model-aggregator-02: the brief's two documents, the free model list, and the
// quality-ranked candidate order.
//
// A trial's brief (the path in MAKEFUN_PROMPT_FILE) is prose with two JSON documents in
// it, found by their `schema` member under this contest's prefix
// (`free-model-aggregator-02/`):
//   - the free model list (`free-model-aggregator-02/models/v1`): one row a model, keyed
//     by the id the gateway routes (`openrouter/<vendor>/<model>:free`), with the house's
//     quality rating in milli-points, what it can do (tools, vision, JSON mode, context
//     length) and its free flag. Only a free model may serve a prompt.
//   - the six prompts (`free-model-aggregator-02/prompts/v1`): the prompts the env sends,
//     in order, each with its id, its api (`openai` or `anthropic`), whether it streams,
//     its weight and the requirements a model must meet. The server reads the requirements
//     off each arriving request body instead (a body that asks for JSON mode or tools
//     names its own requirements), so the manifest is a cross-check, not a dependency.
//
// A brief may spell the model list `models` or `ratings` and the prompts `prompts` or
// `manifest` (the sibling -01 contest's names), and one document may carry both; a
// document is found by the arrays it carries, never by a name alone.
//
// Everything here is deterministic: no clock and no randomness, so `node main.ts --list`
// and the register body build the same document, byte for byte, on a live pass and on
// its replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The house's quality rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this server reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** Find and parse every top-level JSON object in a brief file. */
function extractJsonObjects(text: string): Json[] {
  const objects: Json[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escape = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch === '\\' && inString) {
      escape = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;

    if (ch === '{') {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === '}') {
      depth--;
      if (depth === 0 && start !== -1) {
        const slice = text.slice(start, i + 1);
        try {
          const parsed: unknown = JSON.parse(slice);
          if (isRecord(parsed)) objects.push(parsed);
        } catch {
          // Skip malformed JSON islands.
        }
        start = -1;
      }
    }
  }
  return objects;
}

/** Coerce a loose field value into a boolean or null. */
function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** Read one model row off a raw models array entry. */
function parseModel(raw: unknown): ModelRow | null {
  if (!isRecord(raw) || typeof raw.id !== 'string') return null;
  const id = raw.id.trim();
  if (id === '') return null;
  const rating = typeof raw.rating_milli === 'number' ? raw.rating_milli : typeof raw.rating === 'number' ? Math.round(raw.rating * 1000) : 0;
  const free = raw.free === true || (raw.free === undefined && id.endsWith(':free'));
  return {
    id,
    provider: typeof raw.provider === 'string' && raw.provider !== '' ? raw.provider : 'openrouter',
    display_name: typeof raw.display_name === 'string' && raw.display_name !== '' ? raw.display_name : id,
    rating_milli: rating,
    context_length: typeof raw.context_length === 'number' ? raw.context_length : null,
    tools: boolOrNull(raw.tools),
    vision: boolOrNull(raw.vision),
    json: boolOrNull(raw.json ?? raw.json_mode),
    free,
  };
}

/** Read one prompt row off a raw manifest array entry. */
function parsePrompt(raw: unknown): PromptRow | null {
  if (!isRecord(raw) || typeof raw.id !== 'string') return null;
  const api: 'openai' | 'anthropic' = raw.api === 'anthropic' || raw.wire === 'anthropic' ? 'anthropic' : 'openai';
  const reqObj = isRecord(raw.requires) ? raw.requires : isRecord(raw.requirements) ? raw.requirements : {};
  const requires: Requirement = {};
  if (reqObj.tools === true) requires.tools = true;
  if (reqObj.vision === true) requires.vision = true;
  if (reqObj.json === true || reqObj.json_mode === true) requires.json = true;
  if (typeof reqObj.min_context === 'number') requires.min_context = reqObj.min_context;

  return {
    id: raw.id,
    api,
    stream: raw.stream === true || raw.streaming === true,
    weight: raw.weight === 2 ? 2 : 1,
    requires,
  };
}

/** Read and parse the prompt file; return empty lists when the file is absent. */
export function readBrief(path: string): Brief {
  if (!path) return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE not set' };
  let content: string;
  try {
    content = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `failed to read brief: ${(err as Error).message}` };
  }

  const objects = extractJsonObjects(content);
  let models: ModelRow[] = [];
  let prompts: PromptRow[] = [];

  for (const obj of objects) {
    const schema = typeof obj.schema === 'string' ? obj.schema : '';
    const rawModels = Array.isArray(obj.models) ? obj.models : Array.isArray(obj.ratings) ? obj.ratings : null;
    if (rawModels !== null && (schema === '' || schema.includes(SCHEMA_PREFIX) || schema === MODELS_SCHEMA)) {
      for (const m of rawModels) {
        const parsed = parseModel(m);
        if (parsed !== null && !models.some((existing) => existing.id === parsed.id)) models.push(parsed);
      }
    }

    const rawPrompts = Array.isArray(obj.prompts) ? obj.prompts : Array.isArray(obj.manifest) ? obj.manifest : null;
    if (rawPrompts !== null && (schema === '' || schema.includes(SCHEMA_PREFIX) || schema === PROMPTS_SCHEMA)) {
      for (const p of rawPrompts) {
        const parsed = parsePrompt(p);
        if (parsed !== null && !prompts.some((existing) => existing.id === parsed.id)) prompts.push(parsed);
      }
    }
  }

  return { models, prompts, error: null };
}

/** The free models alone, in the brief's listing order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/**
 * Build the single catalog document registered with the env and printed by --list.
 * It contains exactly the fourteen free models under the single openrouter provider.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const models: CatalogModel[] = freeModels(brief).map((m) => ({
    id: m.id,
    free: true,
    context_length: m.context_length,
  }));

  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

/** Does a model meet every stated requirement of a prompt? */
function matchesRequirements(model: ModelRow, req: Requirement): boolean {
  if (req.tools === true && model.tools !== true) return false;
  if (req.vision === true && model.vision !== true) return false;
  if (req.json === true && model.json !== true) return false;
  if (typeof req.min_context === 'number' && (model.context_length === null || model.context_length < req.min_context)) {
    return false;
  }
  return true;
}

/**
 * Rank candidates for a request:
 * 1. Must be free models.
 * 2. Must satisfy all required capabilities (tools, vision, json, min_context).
 * 3. Sorted by rating_milli DESC (highest quality first).
 * 4. Ties preserve original brief listing order.
 */
export function rankCandidates(brief: Brief, requirements: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((m) => matchesRequirements(m, requirements));
  return eligible.sort((a, b) => {
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return 0;
  });
}
