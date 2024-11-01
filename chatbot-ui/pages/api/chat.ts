import {
  DEFAULT_MAX_TOKENS,
  DEFAULT_SYSTEM_PROMPT,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_K,
} from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';

// @ts-expect-error
import wasm from '../../node_modules/@dqbd/tiktoken/lite/tiktoken_bg.wasm?module';

import tiktokenModel from '@dqbd/tiktoken/encoders/cl100k_base.json';
import { Tiktoken, init } from '@dqbd/tiktoken/lite/init';

export const config = {
  runtime: 'edge',
};

function createPromptFormats(userQuestion: string): string {
  const SYSTEM_START_TOKEN: string = '<<SYS>>';
  const SYSTEM_END_TOKEN: string = '<</SYS>>';
  const INSTRUCTION_START_TOKEN: string = '[INST]';
  const INSTRUCTION_END_TOKEN: string = '[/INST]';
  const SYSTEM_MESSAGE: string = DEFAULT_SYSTEM_PROMPT;

  // const SYSTEM_PROMPT: string = `<s>${INSTRUCTION_START_TOKEN} ${SYSTEM_START_TOKEN}\n${SYSTEM_MESSAGE}\n${SYSTEM_END_TOKEN} ${INSTRUCTION_END_TOKEN}`;
  const SYSTEM_PROMPT = `${SYSTEM_MESSAGE}`;
  const INSTRUCTION_KEY: string = '### Pergunta:';
  const RESPONSE_KEY: string = '### Resposta:';

  const instruction: string = `\n${INSTRUCTION_KEY}\n${userQuestion}`;
  const response: string = `${RESPONSE_KEY}\n`;
  const parts: string[] = [instruction, response].filter((part) => part);

  // const formattedPrompt: string = SYSTEM_PROMPT + '\n\n' + parts.join('\n\n');
  const formattedPrompt: string = parts.join('\n\n'); // removing system prompt for now
  return formattedPrompt;
}

const handler = async (req: Request): Promise<Response> => {
  try {
    const {
      model,
      messages,
      key,
      prompt,
      temperature,
      top_k,
      top_p,
      max_tokens,
      useRag,
    } = (await req.json()) as ChatBody;

    await init((imports) => WebAssembly.instantiate(wasm, imports));
    // const llama_special
    const encoding = new Tiktoken(
      tiktokenModel.bpe_ranks,
      tiktokenModel.special_tokens,
      tiktokenModel.pat_str,
    );

    let promptToSend = prompt;
    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    let top_k_to_use = top_k;
    if (top_k_to_use == null) {
      top_k_to_use = DEFAULT_TOP_K;
    }

    let top_p_to_use = top_p;
    if (top_p_to_use == null) {
      top_p_to_use = DEFAULT_TOP_K;
    }

    let max_new_tokens_to_use = max_tokens;
    if (max_new_tokens_to_use == null) {
      max_new_tokens_to_use = DEFAULT_MAX_TOKENS;
    }

    const prompt_tokens = encoding.encode(promptToSend);

    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];
    let userQuestion = '';
    if (useRag) {
      // get last message
      userQuestion = messages[messages.length - 1].content;
      messages[messages.length - 1].content = createPromptFormats(userQuestion);
    } else {
      for (let i = messages.length - 1; i >= 0; i--) {
        let message = messages[i];
        message.content = createPromptFormats(message.content);
        const tokens = encoding.encode(message.content);

        if (tokenCount + tokens.length + 1000 > model.tokenLimit) {
          break;
        }
        tokenCount += tokens.length;
        messagesToSend = [message, ...messagesToSend];
      }
    }

    console.log('Messages to send: ', messagesToSend);

    encoding.free();

    const stream = await OpenAIStream(
      model,
      promptToSend,
      temperatureToUse,
      top_k_to_use,
      top_p_to_use,
      max_new_tokens_to_use,
      key,
      messagesToSend,
      useRag,
      userQuestion,
    );

    return new Response(stream);
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

export default handler;
