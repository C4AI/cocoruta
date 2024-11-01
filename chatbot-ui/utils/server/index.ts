import { Message } from '@/types/chat';
import { OpenAIModel } from '@/types/openai';

import {
  AZURE_DEPLOYMENT_ID,
  OPENAI_API_HOST,
  OPENAI_API_HOST_RAG,
  OPENAI_API_TYPE,
  OPENAI_API_VERSION,
  OPENAI_ORGANIZATION,
} from '../app/const';

import {
  ParsedEvent,
  ReconnectInterval,
  createParser,
} from 'eventsource-parser';

export class OpenAIError extends Error {
  type: string;
  param: string;
  code: string;

  constructor(message: string, type: string, param: string, code: string) {
    super(message);
    this.name = 'OpenAIError';
    this.type = type;
    this.param = param;
    this.code = code;
  }
}

export const OpenAIStream = async (
  model: OpenAIModel,
  systemPrompt: string,
  temperature: number,
  top_k_to_use: number,
  top_p_to_use: number,
  max_new_tokens_to_use: number,
  key: string,
  messages: Message[],
  useRag: boolean,
  userQuestion?: string,
) => {
  let url = '';

  if (useRag && userQuestion) {
    // url = `${model.rag_host}/v1/chat/completions`;
    url = `${OPENAI_API_HOST}/v1/chat/completions`; // remove rag temporarily
  } else {
    // url = `${model.api_host}/v1/chat/completions`;
    url = `${OPENAI_API_HOST}/generate_stream`; // using TGI generate_stream endpoint because the chat endpoint is not fully compatible with fine-tuned Llama models yet
  }
  if (OPENAI_API_TYPE === 'azure') {
    url = `${OPENAI_API_HOST}/openai/deployments/${AZURE_DEPLOYMENT_ID}/chat/completions?api-version=${OPENAI_API_VERSION}`;
  }

  console.log('model: ', model);
  console.log('URL: ', url);

  const stop_sequences = ['##', ' </s>'];
  const SYSTEM_START_TOKEN = '<<SYS>>';
  const SYSTEM_END_TOKEN = '<</SYS>>';
  const INSTRUCTION_START_TOKEN = '[INST]';
  const INSTRUCTION_END_TOKEN = '[/INST]';
  const special_tokens = [
    SYSTEM_START_TOKEN,
    SYSTEM_END_TOKEN,
    INSTRUCTION_START_TOKEN,
    INSTRUCTION_END_TOKEN,
  ];

  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(OPENAI_API_TYPE === 'openai' && {
        Authorization: `Bearer ${key ? key : process.env.OPENAI_API_KEY}`,
      }),
      ...(OPENAI_API_TYPE === 'azure' && {
        'api-key': `${key ? key : process.env.OPENAI_API_KEY}`,
      }),
      ...(OPENAI_API_TYPE === 'openai' &&
        OPENAI_ORGANIZATION && {
          'OpenAI-Organization': OPENAI_ORGANIZATION,
        }),
    },
    method: 'POST',
    body: JSON.stringify({
      ...(OPENAI_API_TYPE === 'openai' && { model: model.id }),

      inputs: messages.slice(-1)[0].content,
      parameters: {
        max_new_tokens: 512,
        stop: stop_sequences,
        stream: true,
        temperature: 0.3,
        top_p: 0.9,
        top_k: 30,
        repetition_penalty: 1.2,
        do_sample: true,
      },

      // messages: [
      //   // { REMOVE POR ENQUANTO POR CAUSA DO FORMATO DE PROMPT DE SISTEMA DO LLAMA
      //   //   role: 'system',
      //   //   content: systemPrompt,
      //   // },
      //   // ...messages, // send only the last message
      //   ...messages.slice(-1),
      // ],
      // user_question: useRag && userQuestion ? userQuestion : undefined,
      // max_tokens: max_new_tokens_to_use,
      // max_tokens: 512, // temp for red team attack
      // // temperature: temperature,
      // temperature: 0.3, // temp for red team attack
      // // top_p: top_p_to_use,
      // top_p: 0.9, // temp for red team attack
      // // top_k: top_k_to_use,
      // top_k: 30, // temp for red team attack
      // repetition_penalty: 1.2,
      // do_sample: true,
      // stop: stop_sequences,
      // stream: true,
    }),
  });

  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  if (res.status !== 200) {
    const result = await res.json();
    if (result.error) {
      throw new OpenAIError(
        result.error.message,
        result.error.type,
        result.error.param,
        result.error.code,
      );
    } else {
      throw new Error(
        `OpenAI API returned an error: ${
          decoder.decode(result?.value) || result.statusText
        }`,
      );
    }
  }

  function isStopSequence(text: string) {
    return stop_sequences.some((stop_sequence) => text.includes(stop_sequence));
  }

  function isSpecialToken(text: string) {
    return special_tokens.some((special_token) => text.includes(special_token));
  }

  
    

  const stream = new ReadableStream({
    async start(controller) {

      const onParseOpenAI = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === 'event') {
          const data = event.data;
    
          try {
            const json = JSON.parse(data);
            if (json.choices[0].finish_reason != null) {
              controller.close();
              return;
            }

    
            const text = json.choices[0].delta.content;
    
            // check stop_sequences
            if (text && isStopSequence(text)) {
              controller.close();
              return;
            }
    
            //if is special_token, skip it
            if (text && isSpecialToken(text)) {
              return;
            }
    
            const queue = encoder.encode(text);
            controller.enqueue(queue);
          } catch (e) {
            controller.error(e);
          }
        }
      };

      // for TGI /generate_stream, output will be like:
      // data:  {"index":1,"token":{"id":29949,"text":"O","logprob":-0.14685059,"special":false},"generated_text":null,"details":null}?
      // data:  {"index":2,"token":{"id":29875,"text":"i","logprob":0.0,"special":false},"generated_text":null,"details":null}
      // data:  {"index":3,"token":{"id":13,"text":"\n","logprob":-0.2590332,"special":false},"generated_text":null,"details":null}
      // data:  {"index":4,"token":{"id":13,"text":"\n","logprob":0.0,"special":false},"generated_text":null,"details":null}
      // data:  {"index":5,"token":{"id":2277,"text":"##","logprob":0.0,"special":false},"generated_text":"Oi\n\n##","details":null}
      const onParseTGI = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === 'event') {
          const data = event.data;

  
          try {
            const json = JSON.parse(data);
            if (json.generated_text != null) { // the final stream will have the generated_text field filled
              controller.close();
              return;
            }
    
            const text = json.token.text;
    
            const queue = encoder.encode(text);
            controller.enqueue(queue);
          } catch (e) {
            controller.error(e);
          }
        }
      }


      // const parser = createParser(onParseOpenAI);
      const parser = createParser(onParseTGI);

      for await (const chunk of res.body as any) {
        parser.feed(decoder.decode(chunk));
      }
    },
  });

  return stream;
};
