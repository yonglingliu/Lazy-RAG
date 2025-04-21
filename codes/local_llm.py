import lazyllm
llm = lazyllm.TrainableModule('internlm2-chat-7b')
llm.start()
print(llm('你是谁'))