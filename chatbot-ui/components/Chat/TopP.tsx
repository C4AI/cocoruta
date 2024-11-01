import { FC, useContext, useState } from 'react';

import { useTranslation } from 'next-i18next';

import { DEFAULT_TOP_P } from '@/utils/app/const';

import HomeContext from '@/pages/api/home/home.context';

interface Props {
  label: string;
  onChangeTopP: (top_p: number) => void;
}

export const TopPInput: FC<Props> = ({
  label,
  onChangeTopP,
}) => {
  const {
    state: { conversations },
  } = useContext(HomeContext);
  const lastConversation = conversations[conversations.length - 1];
  const [top_p, setTopP] = useState(
    lastConversation?.top_p ?? DEFAULT_TOP_P,
  );
  const { t } = useTranslation('chat');
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value);
    setTopP(newValue);
    onChangeTopP(newValue);
  };

  return (
    // 50% width since top_p will get the other 50%
    <div className="flex flex-col w-[50%]"> 
      <label className="mb-2 text-left text-neutral-700 dark:text-neutral-300 font-bold">
        {label}
      </label>
      {/* <span className="text-[12px] text-black/50 dark:text-white/50 text-sm">
        {t(
          'Higher values like 100 will make the output more random, while lower values like 10 will make it more focused and deterministic.',
        )}
      </span> */}
      {/* <span className="mt-2 mb-1 text-center text-neutral-900 dark:text-neutral-100">
        {top_p.toFixed(1)}
      </span> */}
        <input
            // className="cursor-pointer"
            className='border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500 rounded-md border border-black/10 bg-white shadow-[0_0_10px_rgba(0,0,0,0.10)] dark:border-gray-900/50 dark:bg-[#40414F] dark:text-white dark:shadow-[0_0_15px_rgba(0,0,0,0.10)]'
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={top_p}
            onChange={handleChange}
        />
             
      {/* <input
        className="cursor-pointer"
        type="range"
        min={1}
        max={1000}
        step={10}
        value={top_p}
        onChange={handleChange}
      /> */}
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
