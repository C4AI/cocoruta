import { OPENAI_API_HOST, OPENAI_API_HOST_RAG } from '../utils/app/const';

export interface OpenAIModel {
  id: string;
  name: string;
  maxLength: number; // maximum length of a message
  tokenLimit: number;
  api_host?: string;
  rag_host?: string;
}

export enum OpenAIModelID {
  GPT_3_5 = 'gpt-3.5-turbo',
  GPT_3_5_AZ = 'gpt-35-turbo',
  GPT_4 = 'gpt-4',
  GPT_4_32K = 'gpt-4-32k',
  COCORUTA_7B = 'huggingface/felipeoes/cocoruta-7b',
}

// in case the `DEFAULT_MODEL` environment variable is COCORUTA_7B set or set to an unsupported model
export const fallbackModelID = OpenAIModelID.COCORUTA_7B;

export const OpenAIModels: Record<OpenAIModelID, OpenAIModel> = {
  [OpenAIModelID.GPT_3_5]: {
    id: OpenAIModelID.GPT_3_5,
    name: 'GPT-3.5',
    maxLength: 12000,
    tokenLimit: 4000,
  },
  [OpenAIModelID.GPT_3_5_AZ]: {
    id: OpenAIModelID.GPT_3_5_AZ,
    name: 'GPT-3.5',
    maxLength: 12000,
    tokenLimit: 4000,
  },
  [OpenAIModelID.GPT_4]: {
    id: OpenAIModelID.GPT_4,
    name: 'GPT-4',
    maxLength: 24000,
    tokenLimit: 8000,
  },
  [OpenAIModelID.GPT_4_32K]: {
    id: OpenAIModelID.GPT_4_32K,
    name: 'GPT-4-32K',
    maxLength: 96000,
    tokenLimit: 32000,
  },
  [OpenAIModelID.COCORUTA_7B]: {
    id: OpenAIModelID.COCORUTA_7B,
    name: 'cocoruta-7b',
    maxLength: 8000,
    tokenLimit: 4000,
    api_host: OPENAI_API_HOST,
    rag_host: OPENAI_API_HOST_RAG,

  },
};
