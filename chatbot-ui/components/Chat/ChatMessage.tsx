import {
  IconCheck,
  IconCopy,
  IconEdit,
  IconRobot,
  IconThumbDown,
  IconThumbUp,
  IconTrash,
  IconUser,
  IconX,
} from '@tabler/icons-react';
import { FC, memo, useContext, useEffect, useRef, useState } from 'react';

import { useTranslation } from 'next-i18next';

import { updateConversation } from '@/utils/app/conversation';

import { Message } from '@/types/chat';

import HomeContext from '@/pages/api/home/home.context';

import { CodeBlock } from '../Markdown/CodeBlock';
import { MemoizedReactMarkdown } from '../Markdown/MemoizedReactMarkdown';

import rehypeMathjax from 'rehype-mathjax';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

export interface Props {
  message: Message;
  messageIndex: number;
  onEdit?: (editedMessage: Message) => void;
}

export const ChatMessage: FC<Props> = memo(
  ({ message, messageIndex, onEdit }) => {
    const { t } = useTranslation('chat');

    // check if message.evaluation has another field 'evaluation' and assign it to message.evaluation
    if (message?.evaluation?.evaluation) {
      message.evaluation = message.evaluation.evaluation;
    }

    const {
      state: {
        selectedConversation,
        conversations,
        currentMessage,
        messageIsStreaming,
      },
      dispatch: homeDispatch,
    } = useContext(HomeContext);

    const [isEditing, setIsEditing] = useState<boolean>(false);
    const [isTyping, setIsTyping] = useState<boolean>(false);
    const [messageContent, setMessageContent] = useState(message.content);
    const [messagedCopied, setMessageCopied] = useState(false);

    // evaluation related
    const [evaluation, setEvaluation] = useState({
      inaccurate: message.evaluation?.inaccurate,
      inappropriate: message.evaluation?.inappropriate,
      offensive: message.evaluation?.offensive,
      note: message.evaluation?.note,
      visible: false,
    });

    // update evaluation when message is updated
    useEffect(() => {
      setEvaluation({
        inaccurate: message.evaluation?.inaccurate,
        inappropriate: message.evaluation?.inappropriate,
        offensive: message.evaluation?.offensive,
        note: message.evaluation?.note,
        visible: false,
      });
    }, [selectedConversation, message]);

    // useEffect to update message when evaluation is updated
    useEffect(() => {
      // // update conversation
      // console.log('evaluation', evaluation);
      // console.log('message', message);
    }, [evaluation]);

    const hasEvaluation = () => {
      return (
        evaluation?.inaccurate ||
        evaluation?.inappropriate ||
        evaluation?.offensive
      );
    };

    const handleThumbsDown = async () => {
      setEvaluation({
        ...evaluation,
        visible: !evaluation.visible,
      });
    };

    const getUserIP = async () => {
      const response = await fetch('https://api.ipify.org/?format=json');
      const json = await response.json();
      return json.ip;
    };

    const handleSaveEvaluation = async () => {
      let userIP = await getUserIP();

      setEvaluation((evaluation) => ({
        ...evaluation,
        visible: false,
      }));

      if (!selectedConversation) {
        console.log('selectedConversation not found');
        return;
      }

      const { messages } = selectedConversation;
      const findIndex = messages.findIndex((elm) => elm === message);

      console.log('findIndex', findIndex);
      console.log('message', message);
      console.log("messageIndex", messageIndex);

      // if (findIndex < 0) {
      //   console.log('message not found');
      //   return;
      // }

      // let newMessage = messages[findIndex];
      // newMessage.evaluation = evaluation;
      message.evaluation = evaluation;

      console.log('findIndex', findIndex);
      console.log("messageIndex", messageIndex);
      console.log('message', message);
      // console.log('newMessage', newMessage);


      // messages[findIndex] = newMessage;
      messages[messageIndex] = message;

      const updatedConversation = {
        ...selectedConversation,
        messages,
      };

      // save evaluation to database

      const messagesHistory = selectedConversation.messages.map((message) => {
        return {
          content: message.content,
          role: message.role,
          evaluation: message.evaluation,
        };
      });

      const historyData = {
        conversationId: selectedConversation.id,
        messages: messagesHistory,
        userIP: userIP,
      };

      console.log('historyData', historyData);

      // ngrok server
      await fetch(
        'https://1263-143-107-59-221.ngrok-free.app/save/messages/history',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(historyData),
        },
      )
        .then((response) => {
          console.log('response', response);
        })
        .catch((error) => {
          console.log('error', error);
        });

      console.log('evaluation request', {
        conversationId: selectedConversation.id,
        messageIndex: findIndex,
        userIP: userIP,
        evaluation: evaluation,
      });

      const { single, all } = updateConversation(
        updatedConversation,
        conversations,
      );
      homeDispatch({ field: 'selectedConversation', value: single });
      homeDispatch({ field: 'conversations', value: all });
    };

    console;

    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const toggleEditing = () => {
      setIsEditing(!isEditing);
    };

    const handleInputChange = (
      event: React.ChangeEvent<HTMLTextAreaElement>,
    ) => {
      setMessageContent(event.target.value);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'inherit';
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      }
    };

    const handleEditMessage = () => {
      if (message.content != messageContent) {
        if (selectedConversation && onEdit) {
          onEdit({ ...message, content: messageContent });
        }
      }
      setIsEditing(false);
    };

    const handleDeleteMessage = () => {
      if (!selectedConversation) return;

      const { messages } = selectedConversation;
      const findIndex = messages.findIndex((elm) => elm === message);

      if (findIndex < 0) return;

      if (
        findIndex < messages.length - 1 &&
        messages[findIndex + 1].role === 'assistant'
      ) {
        messages.splice(findIndex, 2);
      } else {
        messages.splice(findIndex, 1);
      }
      const updatedConversation = {
        ...selectedConversation,
        messages,
      };

      const { single, all } = updateConversation(
        updatedConversation,
        conversations,
      );
      homeDispatch({ field: 'selectedConversation', value: single });
      homeDispatch({ field: 'conversations', value: all });
    };

    const handlePressEnter = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !isTyping && !e.shiftKey) {
        e.preventDefault();
        handleEditMessage();
      }
    };

    const copyOnClick = () => {
      if (!navigator.clipboard) return;

      navigator.clipboard.writeText(message.content).then(() => {
        setMessageCopied(true);
        setTimeout(() => {
          setMessageCopied(false);
        }, 2000);
      });
    };

    useEffect(() => {
      setMessageContent(message.content);
    }, [message.content]);

    useEffect(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'inherit';
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      }
    }, [isEditing]);

    return (
      <div
        className={`group md:px-4 ${
          message.role === 'assistant'
            ? 'border-b border-black/10 bg-gray-50 text-gray-800 dark:border-gray-900/50 dark:bg-[#444654] dark:text-gray-100'
            : 'border-b border-black/10 bg-white text-gray-800 dark:border-gray-900/50 dark:bg-[#343541] dark:text-gray-100'
        }`}
        style={{ overflowWrap: 'anywhere' }}
      >
        <div className="relative m-auto flex p-4 text-base md:max-w-2xl md:gap-6 md:py-6 lg:max-w-2xl lg:px-0 xl:max-w-3xl">
          <div className="min-w-[40px] text-right font-bold">
            {message.role === 'assistant' ? (
              // <IconRobot size={30} />
              // svg icon from path
              <svg
                viewBox="293.553 10.453 463.415 344.076"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  fill="white"
                  d="M 701.794 298.538 C 701.061 299.438 700.498 300.244 700.104 300.958 C 699.418 302.211 699.748 302.594 701.094 302.108 C 701.248 302.048 701.508 301.958 701.874 301.838 C 702.188 301.731 702.401 301.834 702.514 302.148 C 702.868 303.134 702.588 303.721 701.674 303.908 C 698.281 304.594 696.571 307.174 696.544 311.648 C 696.544 311.993 696.443 312.328 696.254 312.608 L 695.844 313.248 C 695.531 313.728 695.361 313.681 695.334 313.108 L 694.894 302.318 C 694.873 301.805 694.448 301.402 693.934 301.408 C 668.928 301.788 643.838 301.824 618.664 301.518 C 616.954 301.498 616.244 301.558 614.904 302.728 C 611.944 305.314 608.718 307.524 605.224 309.358 C 594.058 315.224 585.698 318.938 580.144 320.498 C 573.954 322.248 567.664 324.338 561.344 325.108 C 558.618 325.448 555.888 325.728 553.154 325.948 C 538.601 327.128 527.938 327.361 521.164 326.648 C 515.731 326.074 510.304 325.474 504.884 324.848 C 502.124 324.538 499.714 323.498 497.034 323.108 C 494.208 322.701 490.838 321.848 486.924 320.548 C 451.314 308.708 423.044 286.368 400.854 253.628 C 388.601 235.554 380.308 215.831 375.974 194.458 C 375.414 191.668 375.034 187.418 374.334 183.898 C 373.434 179.328 373.394 173.828 373.394 169.188 C 373.401 145.454 373.411 121.721 373.424 97.988 C 373.424 94.718 373.594 90.078 374.294 86.738 C 379.064 64.008 393.404 45.468 416.204 39.518 C 425.104 37.188 434.884 33.378 444.134 33.368 C 462.201 33.341 480.264 33.391 498.324 33.518 C 516.414 33.638 534.644 35.588 551.174 42.588 C 560.354 46.478 569.744 52.368 577.044 59.528 C 583.794 66.158 590.704 77.008 591.994 86.518 C 592.049 86.913 591.743 87.265 591.344 87.268 C 558.311 87.314 525.168 87.354 491.914 87.388 C 489.821 87.388 487.894 87.268 486.134 87.028 C 485.662 86.96 485.24 87.33 485.244 87.808 C 485.271 100.768 485.271 112.771 485.244 123.818 C 485.244 125.458 485.384 126.894 485.664 128.128 C 485.738 128.454 485.911 128.724 486.184 128.938 C 489.824 131.798 493.534 134.551 497.314 137.198 C 512.814 148.068 527.994 159.368 543.464 170.288 C 551.374 175.878 560.024 182.408 568.464 188.288 C 574.964 192.818 582.884 198.938 590.284 203.998 C 608.824 216.658 626.834 230.168 645.324 242.928 C 646.704 243.878 648.004 245.108 649.264 245.978 C 666.511 257.871 683.468 270.168 700.134 282.868 C 702.244 284.478 703.404 286.038 703.844 288.818 C 704.471 292.804 704.624 295.614 704.304 297.248 C 704.104 298.301 703.501 298.661 702.494 298.328 C 702.214 298.241 701.981 298.311 701.794 298.538 Z M 541.604 316.958 C 545.164 316.598 548.534 316.628 551.984 316.008 C 558.064 314.898 564.444 314.528 570.154 313.128 C 572.901 312.448 575.641 311.748 578.374 311.028 C 586.934 308.778 595.474 304.028 603.194 299.758 C 603.554 299.558 603.778 299.338 603.864 299.098 C 604.018 298.698 603.894 298.354 603.494 298.068 C 603 297.71 602.409 297.518 601.804 297.518 C 596.824 297.508 590.764 295.458 587.374 294.628 C 584.854 294.014 582.178 293.184 579.344 292.138 C 568.424 288.111 558.361 283.428 549.154 278.088 C 540.341 272.981 530.664 265.488 520.124 255.608 C 504.998 241.428 493.184 224.908 484.684 206.048 C 483.394 203.168 482.554 199.888 481.464 196.778 C 480.538 194.131 479.858 191.781 479.424 189.728 C 477.474 180.538 474.804 169.878 474.904 161.318 C 475.171 138.151 475.324 114.984 475.364 91.818 C 475.378 82.071 472.914 73.308 467.974 65.528 C 465.261 61.261 461.634 57.534 457.094 54.348 C 445.254 46.038 431.284 43.388 417.304 48.338 C 412.484 50.048 406.544 53.538 402.714 56.668 C 393.574 64.128 387.624 73.421 384.864 84.548 C 383.838 88.708 383.298 95.181 383.244 103.968 C 383.104 126.428 383.148 148.888 383.374 171.348 C 383.744 208.448 401.774 244.258 426.914 271.078 C 439.254 284.248 453.594 295.548 469.604 302.998 C 471.824 304.031 474.038 305.078 476.244 306.138 C 487.064 311.318 498.448 314.528 510.394 315.768 C 520.824 316.858 531.114 318.008 541.604 316.958 Z"
                  data-c-fill="2d1206"
                />
                <ellipse
                  fill="white"
                  transform="matrix(0.9988060593605042, -0.04885000362992287, 0.04885000362992287, 0.9988060593605042, 441.0044860839844, 78.0976333618164)"
                  rx="9.12"
                  ry="8.74"
                  data-c-fill="2d1206"
                />
              </svg>
            ) : (
              // <img
              //   src={'chatbot-ui/public/icons/cocoruta.svg'}
              //   alt="robot"
              //   className="w-7 h-7"
              // />
              <IconUser size={30} />
            )}
          </div>

          <div className="prose mt-[-2px] w-full dark:prose-invert">
            {message.role === 'user' ? (
              <div className="flex w-full">
                {isEditing ? (
                  <div className="flex w-full flex-col">
                    <textarea
                      ref={textareaRef}
                      className="w-full resize-none whitespace-pre-wrap border-none dark:bg-[#343541]"
                      value={messageContent}
                      onChange={handleInputChange}
                      onKeyDown={handlePressEnter}
                      onCompositionStart={() => setIsTyping(true)}
                      onCompositionEnd={() => setIsTyping(false)}
                      style={{
                        fontFamily: 'inherit',
                        fontSize: 'inherit',
                        lineHeight: 'inherit',
                        padding: '0',
                        margin: '0',
                        overflow: 'hidden',
                      }}
                    />

                    <div className="mt-10 flex justify-center space-x-4">
                      <button
                        className="h-[40px] rounded-md bg-blue-500 px-4 py-1 text-sm font-medium text-white enabled:hover:bg-blue-600 disabled:opacity-50"
                        onClick={handleEditMessage}
                        disabled={messageContent.trim().length <= 0}
                      >
                        {t('Salvar & Enviar')}
                      </button>
                      <button
                        className="h-[40px] rounded-md border border-neutral-300 px-4 py-1 text-sm font-medium text-neutral-700 hover:bg-neutral-100 dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800"
                        onClick={() => {
                          setMessageContent(message.content);
                          setIsEditing(false);
                        }}
                      >
                        {t('Cancelar')}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="prose whitespace-pre-wrap flex-1">
                    <div className="prose whitespace-pre-wrap dark:prose-invert flex-1">
                      {message.content}
                    </div>
                  </div>
                )}

                {!isEditing && (
                  <div className="md:-mr-8 ml-1 md:ml-0 flex flex-col md:flex-row gap-4 md:gap-1 items-center md:items-start justify-end md:justify-start">
                    <button
                      className="invisible group-hover:visible focus:visible text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                      onClick={toggleEditing}
                    >
                      <IconEdit size={20} />
                    </button>
                    <button
                      className="invisible group-hover:visible focus:visible text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                      onClick={handleDeleteMessage}
                    >
                      <IconTrash size={20} />
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col">
                <div className="flex flex-row">
                  <MemoizedReactMarkdown
                    className="prose dark:prose-invert flex-1"
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeMathjax]}
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        if (children.length) {
                          if (children[0] == '▍') {
                            return (
                              <span className="animate-pulse cursor-default mt-1">
                                ▍
                              </span>
                            );
                          }

                          children[0] = (children[0] as string).replace(
                            '`▍`',
                            '▍',
                          );
                        }

                        const match = /language-(\w+)/.exec(className || '');

                        return !inline ? (
                          <CodeBlock
                            key={Math.random()}
                            language={(match && match[1]) || ''}
                            value={String(children).replace(/\n$/, '')}
                            {...props}
                          />
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        );
                      },
                      table({ children }) {
                        return (
                          <table className="border-collapse border border-black px-3 py-1 dark:border-white">
                            {children}
                          </table>
                        );
                      },
                      th({ children }) {
                        return (
                          <th className="break-words border border-black bg-gray-500 px-3 py-1 text-white dark:border-white">
                            {children}
                          </th>
                        );
                      },
                      td({ children }) {
                        return (
                          <td className="break-words border border-black px-3 py-1 dark:border-white">
                            {children}
                          </td>
                        );
                      },
                    }}
                  >
                    {`${message.content}${
                      messageIsStreaming &&
                      messageIndex ==
                        (selectedConversation?.messages.length ?? 0) - 1
                        ? '`▍`'
                        : ''
                    }`}
                  </MemoizedReactMarkdown>

                  <div className="md:-mr-8 ml-1 md:ml-0 flex flex-col md:flex-row gap-4 md:gap-1 items-center md:items-start justify-end md:justify-start">
                    {messagedCopied ? (
                      <IconCheck
                        size={20}
                        className="text-green-500 dark:text-green-400"
                      />
                    ) : (
                      <button
                        className="invisible group-hover:visible focus:visible text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                        onClick={copyOnClick}
                      >
                        <IconCopy size={20} />
                      </button>
                    )}
                  </div>
                </div>

                {/* Evaluation */}
                <div className="flex flex-col mt-5">
                  <div className="flex flex-row">
                    <button
                      className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                      onClick={() =>
                        setEvaluation({ ...evaluation, visible: false })
                      }
                    >
                      <IconThumbUp size={24} />
                    </button>

                    {evaluation?.visible || hasEvaluation() ? (
                      <button
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 ml-2"
                        onClick={handleThumbsDown}
                      >
                        <IconThumbDown size={24} color="white" />
                      </button>
                    ) : (
                      <button
                        className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 ml-2"
                        onClick={handleThumbsDown}
                      >
                        <IconThumbDown size={24} />
                      </button>
                    )}
                  </div>
                  <div className="flex flex-row w-full">
                    {evaluation.visible && (
                      <div className="flex flex-col w-full ">
                        <div className="flex flex-row justify-between">
                          <div className="flex flex-col w-full mt-4 mb-2">
                            <label className="mb-2 text-left text-neutral-900 dark:text-neutral-300 font-bold">
                              {t('Por que você escolheu essa avaliação?')}
                            </label>
                          </div>

                          <button
                            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                            onClick={handleThumbsDown}
                          >
                            <IconX size={24} />
                          </button>
                        </div>
                        <div className="flex flex-col">
                          <div className="flex flex-row mr-3 cursor-pointer">
                            <input
                              id="inaccurate"
                              type="checkbox"
                              value=""
                              checked={evaluation.inaccurate}
                              className="w-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600 cursor-pointer"
                              onClick={() =>
                                setEvaluation({
                                  ...evaluation,
                                  inaccurate: !evaluation.inaccurate,
                                })
                              }
                            />
                            <label htmlFor="inaccurate" className="ml-1">
                              {t(
                                'Inacurada (não parece ser uma resposta correta para a pergunta)',
                              )}
                            </label>
                          </div>
                          <div className="flex flex-row mr-3 cursor-pointer">
                            <input
                              id="inappropriate"
                              type="checkbox"
                              value=""
                              checked={evaluation.inappropriate}
                              className="w-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600 cursor-pointer"
                              onClick={() =>
                                setEvaluation({
                                  ...evaluation,
                                  inappropriate: !evaluation.inappropriate,
                                })
                              }
                            />
                            <label htmlFor="inappropriate" className="ml-1">
                              {t(
                                'Inapropriada (não pertinente ao contexto jurídico da Amazônia Azul)',
                              )}
                            </label>
                          </div>
                          <div className="flex flex-row cursor-pointer">
                            <input
                              id="offensive"
                              type="checkbox"
                              value=""
                              checked={evaluation.offensive}
                              className="w-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:bg-gray-700 dark:border-gray-600 cursor-pointer"
                              onClick={() =>
                                setEvaluation({
                                  ...evaluation,
                                  offensive: !evaluation.offensive,
                                })
                              }
                            />
                            <label htmlFor="offensive" className="ml-1">
                              {t(
                                'Ofensiva (contém discurso de ódio, machismo, LGBTfobia, racismo, etc.)',
                              )}
                            </label>
                          </div>
                        </div>
                        <div className="flex flex-row w-full mt-2">
                          <textarea
                            // className="w-full h-20 mt-4 resize-none border-none dark:bg-[#343541]"
                            className="w-full h-20 mt-4 resize-none rounded-lg border border-neutral-200 bg-transparent px-4 py-3 text-neutral-900 dark:border-neutral-800 dark:text-neutral-100"
                            placeholder="Adicione algum comentário (opcional)"
                            value={evaluation.note}
                            onChange={(e) =>
                              setEvaluation({
                                ...evaluation,
                                note: e.target.value,
                              })
                            }
                            style={{
                              fontFamily: 'inherit',
                              fontSize: 'inherit',
                              lineHeight: 'inherit',
                              overflow: 'auto',
                            }}
                          />
                        </div>
                        <div className="flex flex-row">
                          <button
                            className="h-[40px] mt-4 rounded-md bg-blue-500 px-4 py-1 text-sm font-medium text-white enabled:hover:bg-blue-600 disabled:opacity-50"
                            onClick={
                              handleSaveEvaluation
                              // () =>
                              // setEvaluation({
                              //   ...evaluation,
                              //   visible: false,
                              // })
                            }
                            disabled={
                              !evaluation.inaccurate &&
                              !evaluation.inappropriate &&
                              !evaluation.offensive
                            }
                          >
                            {t('Enviar')}
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  },
);
ChatMessage.displayName = 'ChatMessage';
