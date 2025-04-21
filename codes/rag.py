import lazyllm

prompt = (
    'You will act as an AI question-answering assistant and complete a dialogue task.'  
    'In this task, you need to provide your answers based on the given context and questions.'  
)
documents = lazyllm.Document(dataset_path="doc")

with lazyllm.pipeline() as ppl:
    ppl.retriever = lazyllm.Retriever(
        doc=documents, group_name="CoarseChunk", similarity="bm25_chinese",
        topk=3, output_format="content", join='')
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | lazyllm.bind(query=ppl.input) 
    ppl.llm = lazyllm.OnlineChatModule('DeepSeek-V3', source='sensenova').prompt(lazyllm.ChatPrompter(instruction=prompt, extro_keys=['context_str']))

query = input("请输入您的问题\n")
res = ppl(query)
print(f"With RAG Answer:\n{res}")
