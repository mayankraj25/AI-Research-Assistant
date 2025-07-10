from langchain.document_loaders import WebBaseLoader

def load_url(url):
    loader=WebBaseLoader(url)
    return loader.load()