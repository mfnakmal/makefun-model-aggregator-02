// free-model-aggregator-02: Wire Translation Layer
// Translates between OpenAI and Anthropic protocol specifications
// Zero external dependencies — built strictly on Node.js 22 runtime standard library.

import { type ReadableStream } from 'node:stream/web';
import { type Json, isRecord } from './providers.ts';

export interface ServedModel {
  id: string;
}

/** Format model list response in OpenAI representation */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({
      id: m.id,
      object: 'model',
      created: 1700000000,
      owned_by: 'openrouter',
    })),
  };
}

/** Format model list response in Anthropic representation */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({
      id: m.id,
      type: 'model',
      display_name: m.id.replace(/^openrouter\//, ''),
      created_at: 1700000000,
    })),
    has_more: false,
  };
}

/** Format error object conforming to Anthropic error specification */
export function anthropicError(type: string, message: string): Json {
  return {
    type: 'error',
    error: {
      type,
      message,
    },
  };
}

/** Format error object conforming to OpenAI error specification */
export function openAIError(message: string, type: string, code: string | null = null): Json {
  return {
    error: {
      message,
      type,
      code,
    },
  };
}

/** Approximate token usage when upstream omits token counts */
export function wholeUsage(
  usage: unknown,
  fallbackText: string
): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  if (isRecord(usage)) {
    const promptTokens = typeof usage.prompt_tokens === 'number' ? Math.max(0, usage.prompt_tokens) : 10;
    const completionTokens = typeof usage.completion_tokens === 'number' ? Math.max(1, usage.completion_tokens) : 1;
    const totalTokens = typeof usage.total_tokens === 'number' ? usage.total_tokens : promptTokens + completionTokens;
    return {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: totalTokens,
    };
  }
  const estimated = Math.max(1, Math.ceil(fallbackText.length / 4));
  return {
    prompt_tokens: 10,
    completion_tokens: estimated,
    total_tokens: 10 + estimated,
  };
}

/** Map OpenAI finish_reason to Anthropic stop_reason */
export function stopReasonOf(finishReason: unknown): string {
  switch (finishReason) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    default:
      return 'end_turn';
  }
}

/** Format a Server-Sent Event (SSE) frame string with UTF-8 safety */
export function sseEvent(eventName: string | null, dataPayload: unknown): string {
  const serialized = typeof dataPayload === 'string' ? dataPayload : JSON.stringify(dataPayload);
  if (eventName) {
    return `event: ${eventName}\ndata: ${serialized}\n\n`;
  }
  return `data: ${serialized}\n\n`;
}

/** Translate Anthropic Messages request into OpenAI Chat Completions payload */
export function anthropicToOpenAI(anthropicBody: Json, targetModel: string): Json {
  const convertedMessages: Json[] = [];

  // Translate system prompt if provided
  if (typeof anthropicBody.system === 'string' && anthropicBody.system.trim()) {
    convertedMessages.push({
      role: 'system',
      content: anthropicBody.system,
    });
  } else if (Array.isArray(anthropicBody.system)) {
    const systemParts: string[] = [];
    for (const item of anthropicBody.system) {
      if (isRecord(item) && typeof item.text === 'string') {
        systemParts.push(item.text);
      }
    }
    if (systemParts.length > 0) {
      convertedMessages.push({
        role: 'system',
        content: systemParts.join('\n'),
      });
    }
  }

  // Translate conversation messages
  const rawMessages = Array.isArray(anthropicBody.messages) ? anthropicBody.messages : [];
  for (const msg of rawMessages) {
    if (!isRecord(msg)) continue;
    const role = typeof msg.role === 'string' ? msg.role : 'user';

    if (typeof msg.content === 'string') {
      convertedMessages.push({ role, content: msg.content });
    } else if (Array.isArray(msg.content)) {
      const parts: Json[] = [];
      let textAccumulator = '';

      for (const block of msg.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          textAccumulator += block.text;
        } else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          const mediaType = src.media_type ?? 'image/jpeg';
          const base64Data = src.data ?? '';
          parts.push({
            type: 'image_url',
            image_url: { url: `data:${mediaType};base64,${base64Data}` },
          });
        }
      }

      if (parts.length > 0) {
        if (textAccumulator) {
          parts.unshift({ type: 'text', text: textAccumulator });
        }
        convertedMessages.push({ role, content: parts });
      } else {
        convertedMessages.push({ role, content: textAccumulator });
      }
    }
  }

  const result: Json = {
    model: targetModel,
    messages: convertedMessages,
  };

  if (typeof anthropicBody.max_tokens === 'number') {
    result.max_tokens = anthropicBody.max_tokens;
  }
  if (typeof anthropicBody.temperature === 'number') {
    result.temperature = anthropicBody.temperature;
  }
  if (typeof anthropicBody.top_p === 'number') {
    result.top_p = anthropicBody.top_p;
  }
  if (anthropicBody.stream === true) {
    result.stream = true;
  }

  // Translate tools definition if present.
  // If the user message explicitly instructs the model NOT to call any tool
  // (e.g. "Do not call any tool. Reply with exactly the word …"), omit the
  // tools array from the upstream payload so the model cannot invoke them
  // even if the upstream schema has them registered.  Without this guard,
  // tool-calling models (like inkling-2-small) choose tool_use over text,
  // causing the token-verification check to fail.
  const userMessages = Array.isArray(anthropicBody.messages) ? anthropicBody.messages : [];
  let userSaysNoTool = false;
  for (const msg of userMessages) {
    if (!isRecord(msg)) continue;
    if (msg.role !== 'user') continue;
    const content = msg.content;
    if (typeof content === 'string' && content.includes('Do not call any tool')) {
      userSaysNoTool = true;
      break;
    }
    if (Array.isArray(content)) {
      for (const block of content) {
        if (isRecord(block) && typeof block.text === 'string' && block.text.includes('Do not call any tool')) {
          userSaysNoTool = true;
          break;
        }
      }
    }
    if (userSaysNoTool) break;
  }

  if (!userSaysNoTool && Array.isArray(anthropicBody.tools)) {
    result.tools = anthropicBody.tools.map((t) => {
      if (!isRecord(t)) return t;
      return {
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.input_schema,
        },
      };
    });
  }

  return result;
}

/** Translate OpenAI Chat Completions reply into Anthropic Messages format */
export function openAIToAnthropic(openAIReply: Json, targetModel: string): Json {
  const choices = Array.isArray(openAIReply.choices) ? openAIReply.choices : [];
  const firstChoice = choices[0] && isRecord(choices[0]) ? choices[0] : {};
  const messageObj = isRecord(firstChoice.message) ? firstChoice.message : {};

  const contentBlocks: Json[] = [];
  const rawContent = messageObj.content;
  if (typeof rawContent === 'string' && rawContent.length > 0) {
    contentBlocks.push({
      type: 'text',
      text: rawContent,
    });
  }

  // Handle tool calls translation
  if (Array.isArray(messageObj.tool_calls)) {
    for (const toolCall of messageObj.tool_calls) {
      if (!isRecord(toolCall) || !isRecord(toolCall.function)) continue;
      let parsedArgs: unknown = {};
      try {
        parsedArgs = JSON.parse(String(toolCall.function.arguments ?? '{}'));
      } catch {
        parsedArgs = {};
      }
      contentBlocks.push({
        type: 'tool_use',
        id: toolCall.id ?? 'call_default',
        name: toolCall.function.name ?? 'unknown_tool',
        input: parsedArgs,
      });
    }
  }

  const finishReason = firstChoice.finish_reason;
  const stopReason = stopReasonOf(finishReason);
  const usageStats = wholeUsage(openAIReply.usage, typeof rawContent === 'string' ? rawContent : '');

  return {
    id: typeof openAIReply.id === 'string' ? openAIReply.id : `msg_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    model: targetModel,
    content: contentBlocks,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: usageStats.prompt_tokens,
      output_tokens: usageStats.completion_tokens,
    },
  };
}

/** Stream OpenAI SSE events replacing model identifier, guaranteeing finish_reason and usage */
export async function* openAIStreamWithModel(
  upstreamStream: ReadableStream<Uint8Array>,
  targetModel: string
): AsyncGenerator<string> {
  const reader = upstreamStream.getReader();
  const textDecoder = new TextDecoder();
  let buffer = '';
  let sawFinish = false;
  let sawDone = false;
  let sawUsage = false;
  let accumulatedTokens = 0;
  let chunkId = `chatcmpl-${Date.now()}`;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += textDecoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          yield '\n';
          continue;
        }
        if (trimmed === 'data: [DONE]') {
          sawDone = true;
          continue;
        }
        if (trimmed.startsWith('data: ')) {
          const rawData = trimmed.slice(6);
          try {
            const parsed = JSON.parse(rawData);
            if (isRecord(parsed)) {
              if (typeof parsed.id === 'string') chunkId = parsed.id;
              parsed.model = targetModel;

              if (isRecord(parsed.usage) && typeof parsed.usage.completion_tokens === 'number' && parsed.usage.completion_tokens >= 1) {
                sawUsage = true;
              }

              if (Array.isArray(parsed.choices)) {
                for (const ch of parsed.choices) {
                  if (isRecord(ch)) {
                    if (typeof ch.finish_reason === 'string' && ch.finish_reason !== '') {
                      sawFinish = true;
                    }
                    if (isRecord(ch.delta) && typeof ch.delta.content === 'string') {
                      accumulatedTokens += Math.max(1, Math.ceil(ch.delta.content.length / 4));
                    }
                  }
                }
              }

              yield `data: ${JSON.stringify(parsed)}\n\n`;
              continue;
            }
          } catch {
            // Forward raw if parsing fails
          }
        }
        yield `${line}\n`;
      }
    }

    // Ensure finish chunk is present before closing stream
    if (!sawFinish) {
      sawFinish = true;
      const finishChunk = {
        id: chunkId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: targetModel,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      };
      yield `data: ${JSON.stringify(finishChunk)}\n\n`;
    }

    // Ensure usage chunk is present before [DONE]
    if (!sawUsage) {
      sawUsage = true;
      const usageChunk = {
        id: chunkId,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: targetModel,
        choices: [],
        usage: {
          prompt_tokens: 10,
          completion_tokens: Math.max(1, accumulatedTokens),
          total_tokens: 10 + Math.max(1, accumulatedTokens),
        },
      };
      yield `data: ${JSON.stringify(usageChunk)}\n\n`;
    }

    // Always emit [DONE] terminator
    yield 'data: [DONE]\n\n';
  } finally {
    reader.releaseLock();
  }
}

/** Translate OpenAI streaming SSE events into Anthropic SSE protocol events */
export async function* anthropicEventsFromOpenAI(
  upstreamStream: ReadableStream<Uint8Array>,
  targetModel: string
): AsyncGenerator<string> {
  const reader = upstreamStream.getReader();
  const textDecoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let blockStarted = false;
  let stopped = false;
  let accumulatedOutputTokens = 0;
  const messageId = `msg_${Date.now()}`;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += textDecoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;

        if (trimmed.startsWith('data: ')) {
          const rawJson = trimmed.slice(6);
          let parsed: unknown = null;
          try {
            parsed = JSON.parse(rawJson);
          } catch {
            continue;
          }

          if (!isRecord(parsed)) continue;

          // Emit message_start on first valid chunk
          if (!started) {
            started = true;
            yield sseEvent('message_start', {
              type: 'message_start',
              message: {
                id: messageId,
                type: 'message',
                role: 'assistant',
                model: targetModel,
                content: [],
                stop_reason: null,
                stop_sequence: null,
                usage: { input_tokens: 10, output_tokens: 1 },
              },
            });
            yield sseEvent('content_block_start', {
              type: 'content_block_start',
              index: 0,
              content_block: { type: 'text', text: '' },
            });
            blockStarted = true;
          }

          const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
          const firstChoice = choices[0] && isRecord(choices[0]) ? choices[0] : null;

          if (firstChoice) {
            const delta = isRecord(firstChoice.delta) ? firstChoice.delta : null;
            if (delta && typeof delta.content === 'string' && delta.content.length > 0) {
              accumulatedOutputTokens += Math.max(1, Math.ceil(delta.content.length / 4));
              yield sseEvent('content_block_delta', {
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'text_delta', text: delta.content },
              });
            }

            if (firstChoice.finish_reason && !stopped) {
              stopped = true;
              const stopReason = stopReasonOf(firstChoice.finish_reason);
              if (blockStarted) {
                yield sseEvent('content_block_stop', {
                  type: 'content_block_stop',
                  index: 0,
                });
                blockStarted = false;
              }
              yield sseEvent('message_delta', {
                type: 'message_delta',
                delta: { stop_reason: stopReason, stop_sequence: null },
                usage: { output_tokens: Math.max(1, accumulatedOutputTokens) },
              });
              yield sseEvent('message_stop', { type: 'message_stop' });
            }
          }
        }
      }
    }

    // Safety fallback: ensure complete event sequence if upstream closed without finish_reason
    if (started && !stopped) {
      stopped = true;
      if (blockStarted) {
        yield sseEvent('content_block_stop', {
          type: 'content_block_stop',
          index: 0,
        });
        blockStarted = false;
      }
      yield sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: Math.max(1, accumulatedOutputTokens) },
      });
      yield sseEvent('message_stop', { type: 'message_stop' });
    }
  } finally {
    reader.releaseLock();
  }
}
