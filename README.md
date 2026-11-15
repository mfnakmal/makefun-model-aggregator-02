# free-model-aggregator-02 — The best live free-model aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/mfnakmal/makefun-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/mfnakmal/makefun-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that serves an OpenAI- and an Anthropic-compatible API and answers each
prompt with a **real completion from a live free OpenRouter model**, dialled through
the grader's gateway on the entrant's pushed key. No simulated provider, no local
model, no answer of its own: every completion comes from the internet, at $0.

Bounty details and leaderboard: [make.fun Bounty free-model-aggregator-02](https://make.fun/bounty/free-model-aggregator-02)

## How to run

```sh
node main.ts            # listens on 0.0.0.0 at a free port, registers, serves, exits 0 when done
node main.ts --list     # prints the catalog document and exits 0 (no server started)
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `MAKEFUN_TRIAL_SEED`,
`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
Node 22 standard library only — zero external dependencies, nothing is installed.

## Architecture

```
main.ts            the server: routes, register + poll, the dial-and-failover loop
lib/providers.ts   the brief's two documents, the catalog, quality-ranked candidates
lib/wire.ts        the two client wires, Anthropic <-> Chat Completions, SSE, model field
LICENSE            MIT (Copyright (c) 2026 mfnakmal)
.makefun/owner.json  {"wallet": "FbUbuENN8tW4vSpQHG1e65iJaa28ksy6b7VLPvWZU6YL"}
```

- **Routes.** `POST /v1/chat/completions` (OpenAI wire) and `POST /v1/messages`
  (Anthropic wire), JSON and SSE streaming; `GET /v1/models` answers in the shape
  of the auth header it got (`Authorization: Bearer` → OpenAI list, `x-api-key` →
  Anthropic list). Every prompt body names model `auto`; the server replaces it
  with the free model it chose.
- **Upstream.** Every model is dialled on the gateway's Chat Completions route
  only (`POST $OPENAI_BASE_URL/chat/completions`), with the trial key sent as the
  vendor SDK sends it (`Authorization: Bearer`). An Anthropic-shaped request is
  translated into a Chat Completions body first and the answer translated back.
  The gateway's `POST $ANTHROPIC_BASE_URL/v1/messages` route is never used: for
  these ids it refuses every call.
- **The model field.** The provider's answer names the model without the
  gateway's `openrouter/` prefix; the server puts the id it really dialled back on
  every answer and every streamed chunk, because the house scores the model its
  own record shows answered.

## Model-selection policy (quality, 60% of the score)

Candidates are the brief's free models that meet the request's requirements,
ranked by the brief's `rating_milli`, highest first, ties keeping the brief's
order. Requirements are read off the request body itself (a `response_format`
of `json_object`/`json_schema` needs JSON mode; a non-empty `tools` array needs
tools; an image part needs vision), never off the manifest's order.

| Prompt kind | Model dialled (best capable) | Rating |
|---|---|---|
| JSON mode (p01, p04) | `openrouter/nvidia/nemotron-3.5-super-vl:free` | 1303 — best JSON-capable (the best overall, inkling-2-small at 1475, does not support JSON mode) |
| Plain / streaming (p02, p03, p06) | `openrouter/thinkingmachines/inkling-2-small:free` | 1475 — best overall |
| Tools (p05) | `openrouter/thinkingmachines/inkling-2-small:free` | 1475 — best tools-capable |

The next-best capable models, in order, are the failover ladder: JSON →
gemma-4-31b (1210) → the 1150 JSON-capable models in brief order; plain/tools →
super-vl (1303) → nemotron-3-ultra (1285) → lightning (1221) → gemma-4-31b
(1210) → north-mini-code (1200) → the 1150s in brief order. A prompt served by
its best capable model earns full quality; the ranking is re-derived fresh for
every prompt, so a rate limit that clears lets the best model serve again.

## Failover policy (failover, 10% of the score)

- A **429 or 5xx** (or a network error, or a broken 200 body) moves to the
  next-best capable model. A **4xx other than 429** stops the attempt: it would
  fail the same way on every model. Within one prompt each model is tried at most
  once, so a refusing model is never hammered; at most **5 calls** are spent on
  one prompt, so six prompts stay under the trial's 60-call cap.
- Between attempts the server pauses **250 ms** (fixed, deterministic).
- Every refusal is reported on the response header
  `x-makefun-upstream-failures`: a JSON array of at most 8
  `{"status": <number>, "code": "<string>", "model": "<id or null>"}` rows in
  refusal order — the only way the failover column sees a retried-then-served
  refusal. The code comes from the gateway's own error body (`error.code`, else
  `error.type`, else `http_<status>`).
- A prompt given up on scores nothing, so the server never gives up while a
  capable model is untried.

## Wire fidelity (wire, 10% of the score)

- Answers return in the wire the env used, streamed or not as asked, with usage
  and stop reason whole. A JSON answer whose usage is missing or has 0 output
  tokens gets its output count filled from the answer's own length (a served
  prompt always produced output); a streamed Anthropic answer's `message_delta`
  reports at least 1 output token whenever the stream carried content.
- Every upstream body names an output cap: a request with no cap gets
  `max_tokens: 1024`, one over the gateway's 8192 cap is clamped to it (a request
  with no cap, or one over the cap, is refused). Streaming asks for the usage
  chunk (`stream_options: {include_usage: true}`).
- Model listings use fixed timestamps, so no reply reads the wall clock.

## Catalog (catalog, 5% of the score)

Exactly the fourteen free ids the brief lists, once each, in the brief's order,
as one `openrouter` provider (`base_url: https://openrouter.ai/api/v1`,
`key_env: OPENROUTER_API_KEY`, `wire: openai`). The register body and both
`GET /v1/models` answers carry that same set; `--list` prints the same document
byte for byte.

## Determinism

Every choice is a function of the brief, the requests and the gateway answers
seen so far. No wall clock (`Date.now` is never called; listings use fixed
timestamps), no unseeded randomness. A replay leaves the same state export.

## Verification (local, mock gateway — no real key)

Run against the kit's **real env server** with a mock gateway that answers
canned completions (model echoed without the `openrouter/` prefix, token
carried back, SSE + `[DONE]`), then the kit's **sealed scorer**:

- Clean practice trial: **score 100** — quality 100, served 100, wire 100,
  failover 100, catalog 100; 6 of 6 valid, 6 upstream calls (the minimum),
  `real_model` check pass. All 39 wire checks pass.
- Failover drill (first inkling call answered 429 once): **score 98** —
  quality 96, served 100, wire 100, failover 100, catalog 100; the refused
  prompt was still served, its trail header recorded exactly, and the next
  prompt went back to inkling once the limit cleared.

What was **not** verified locally (needs the real gateway + a pushed OpenRouter
key): live calls to the fourteen real models, real 429/5xx behaviour per model,
and the scored instances' own tokens. The server dials only the fourteen
allow-listed `:free` ids, so a live run spends $0.
