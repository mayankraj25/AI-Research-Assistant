import streamlit as st
from main import load_all_documents, build_vectorstore, build_memory_chain
from utils.summarizer import summarize_chunks
from dotenv import load_dotenv
load_dotenv()
import tempfile

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

st.set_page_config(page_title="AI ResearchAsistant",page_icon="🤖",layout="wide")
st.title("AI Research Assistant  🤖")

st.sidebar.header("Upload Your Documents")
pdfs=st.sidebar.file_uploader("Upload your pdf files",type=["pdf"],accept_multiple_files=True)
urls=st.sidebar.text_area("Enter URLs (comma seperated)",placeholder="https://example.com, https://another-example.com")

sources={
    "pdfs":[save_uploaded_file(pdf) for pdf in pdfs] if pdfs else [],
    "urls":urls.split(",") if urls else []
}


if st.sidebar.button("Load Document"):
    with st.spinner("Loading documnets.."):
        docs=load_all_documents(sources)
        if not docs:
            st.error("No documents found. Please upload PDFs or enter valid URLs.")
        else:
            st.success(f"Loaded {len(docs)} documents successfully.")
            vectorstore,chunk=build_vectorstore(docs)
            st.session_state["vectorstore"] = vectorstore
            st.session_state["chunks"] = chunk
            qa_chain=build_memory_chain(vectorstore)
            st.session_state.qa_chain=qa_chain
            st.success("✅ Documnets processed successfully. You can start chating now")
            

if "chunks" in st.session_state and st.button("Summarize"):
    with st.spinner("📝 Summarizing..."):
        summary=summarize_chunks(st.session_state["chunks"])
        st.subheader("Summary of Documents")
        st.write(summary)

if "qa_chain" in st.session_state:
    st.subheader("Chat with your documents")
    user_query=st.text_input("Ask a question about your documents:")
    if user_query:
        with st.spinner("Generating response..."):
            response=st.session_state.qa_chain.invoke({"question":user_query})
            st.write(response["answer"])



