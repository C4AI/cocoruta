import { OpenAIModel } from './openai';

export interface Message {
  evaluation: {
    evaluation?: any; inaccurate: boolean; inappropriate: boolean; offensive: boolean; note: string; visible: boolean; 
};
  role: Role;
  content: string;
}

export type Role = 'assistant' | 'user';

export interface ChatBody {
  model: OpenAIModel;
  messages: Message[];
  key: string;
  prompt: string;
  temperature: number;
  top_k: number;
  top_p: number;
  max_tokens: number;
  useRag: boolean;
}

export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  model: OpenAIModel;
  prompt: string;
  temperature: number;
  top_k: number;
  top_p: number;
  max_tokens: number;
  folderId: string | null;
  useRag: boolean;
}
