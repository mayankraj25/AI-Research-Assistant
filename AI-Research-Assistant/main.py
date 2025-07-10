from utils.pdf_loader import load_pdf
from utils.web_loader import load_url
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_all_documents(source: dict):
    all_docs=[]
    if source.get("pdfs"):
        for pdf in source["pdfs"]:
            all_docs.extend(load_pdf(pdf))
    if source.get("websites"):
        for url in source["websites"]:
            all_docs.extend(load_url(url))
    return all_docs

def build_vectorstore(docs):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunk=text_splitter.split_documents(docs)
    embeddings=OpenAIEmbeddings()
    return Chroma.from_documents(chunk,embeddings),chunk

def build_memory_chain(vectorstore):
    llm=ChatOpenAI(model_name="gpt-4o-mini",temperature=0.1)
    retriever=vectorstore.as_retriever(search_kwargs={"k":4})
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=memory)
        


