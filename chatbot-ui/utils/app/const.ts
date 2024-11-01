console.log("Environment variables:" + JSON.stringify(process.env, null, 2));

export const DEFAULT_SYSTEM_PROMPT =
  process.env.NEXT_PUBLIC_DEFAULT_SYSTEM_PROMPT ||
  `Você é um respondedor de perguntas, especializado em responder perguntas sobre a legislação da Amazônia Azul. Você deve responder a pergunta abaixo, fornecendo uma resposta completa e detalhada. Suas respostas não devem incluir conteúdo prejudicial, como discurso de ódio, linguagem ofensiva ou conteúdo sexualmente explícito.`;

export const OPENAI_API_HOST =
  process.env.OPENAI_API_HOST;

export const OPENAI_API_HOST_RAG =
  process.env.OPENAI_API_HOST_RAG;

export const DEFAULT_TEMPERATURE = parseFloat(
  process.env.NEXT_PUBLIC_DEFAULT_TEMPERATURE || '0.3',
);

export const DEFAULT_MAX_TOKENS = parseInt(
  process.env.NEXT_PUBLIC_DEFAULT_MAX_TOKENS || '512',
);

export const DEFAULT_TOP_K = parseInt(
  process.env.NEXT_PUBLIC_DEFAULT_TOP_K || '30',
);
export const DEFAULT_TOP_P = parseFloat(
  process.env.NEXT_PUBLIC_DEFAULT_TOP_P || '0.9',
);

export const OPENAI_API_TYPE = process.env.OPENAI_API_TYPE || 'openai';

export const OPENAI_API_VERSION =
  process.env.OPENAI_API_VERSION || '2023-03-15-preview';

export const OPENAI_ORGANIZATION = process.env.OPENAI_ORGANIZATION || '';

export const AZURE_DEPLOYMENT_ID = process.env.AZURE_DEPLOYMENT_ID || '';
