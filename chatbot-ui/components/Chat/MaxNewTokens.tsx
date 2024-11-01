import { FC, useContext, useState } from 'react';

import { useTranslation } from 'next-i18next';

import { DEFAULT_MAX_TOKENS } from '@/utils/app/const';

import HomeContext from '@/pages/api/home/home.context';

interface Props {
  label: string;
  onChangeMaxNewTokens: (max_tokens: number) => void;
}

export const MaxNewTokensInput: FC<Props> = ({
  label,
  onChangeMaxNewTokens,
}) => {
  const {
    state: { conversations },
  } = useContext(HomeContext);
  const lastConversation = conversations[conversations.length - 1];
  const [max_tokens, setMaxNewTokens] = useState(
    lastConversation?.max_tokens ?? DEFAULT_MAX_TOKENS,
  );
  const { t } = useTranslation('chat');
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value);
    setMaxNewTokens(newValue);
    onChangeMaxNewTokens(newValue);
  };

  return (
    <div className="flex flex-col">
      <label className="mb-2 text-left text-neutral-700 dark:text-neutral-300 font-bold">
        {label}
      </label>
      <span className=" bg-transparent text-neutral-900 dark:text-neutral-100 mb-4">
        {t(
          'Limita o número de tokens gerados. Um valor mais alto pode resultar em uma resposta mais longa.',
        )}
      </span>
      <input
        className="border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500 rounded-md border border-black/10 bg-white shadow-[0_0_10px_rgba(0,0,0,0.10)] dark:border-gray-900/50 dark:bg-[#40414F] dark:text-white dark:shadow-[0_0_15px_rgba(0,0,0,0.10)]"
        min={0}
        value={max_tokens}
        onChange={handleChange}
      />
      {/* <ul className="w mt-2 pb-8 flex justify-between px-[24px] text-neutral-900 dark:text-neutral-100">
        <li className="flex justify-center">
          <span className="absolute">{t('Precise')}</span>
        </li>
        <li className="flex justify-center">
          <span className="absolute">{t('Neutral')}</span>
        </li>
        <li className="flex justify-center">
          <span className="absolute">{t('Creative')}</span>
        </li>
      </ul> */}
    </div>
  );
};
