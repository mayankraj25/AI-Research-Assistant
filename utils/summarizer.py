from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

def summarize_chunks(chunks):
    docs=chunks
    llm=ChatOpenAI(model_name="gpt-4o-mini",temperature=0.1)
    chain=load_summarize_chain(llm,chain_type="map_reduce")
    summary=chain.run(docs)
    return summary