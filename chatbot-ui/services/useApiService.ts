import { useCallback } from 'react';

import { useFetch } from '@/hooks/useFetch';
import { OPENAI_API_HOST, OPENAI_API_HOST_RAG } from '@/utils/app/const';

export interface GetModelsRequestProps {
  key: string;
}

const useApiService = () => {
  const fetchService = useFetch();

  // const getModels = useCallback(
  // 	(
  // 		params: GetManagementRoutineInstanceDetailedParams,
  // 		signal?: AbortSignal
  // 	) => {
  // 		return fetchService.get<GetManagementRoutineInstanceDetailed>(
  // 			`/v1/ManagementRoutines/${params.managementRoutineId}/instances/${params.instanceId
  // 			}?sensorGroupIds=${params.sensorGroupId ?? ''}`,
  // 			{
  // 				signal,
  // 			}
  // 		);
  // 	},
  // 	[fetchService]
  // );

  const getModels = useCallback(
    (params: GetModelsRequestProps, signal?: AbortSignal) => {
      const models = [
        {
          id: 'huggingface/felipeoes/cocoruta-7b',
          name: 'cocoruta-7b',
          api_host: OPENAI_API_HOST,
          rag_host: OPENAI_API_HOST_RAG,
        },
      ];
      return models;
    },
    [fetchService],
  );

  return {
    getModels,
  };
};

export default useApiService;
