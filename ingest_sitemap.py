from langchain_community.document_loaders.sitemap import SitemapLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma

OPENAI_API_TOKEN = os.getenv('OPENAI_API_KEY')

print("start")
text_splitter = RecursiveCharacterTextSplitter()
sitemap_loader = SitemapLoader(web_path="site.xml", is_local=True)
print("loading")
docs = sitemap_loader.load_and_split(text_splitter)
print("loaded")
print(docs)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_TOKEN, model="text-embedding-3-large")

if not os.path.isdir("embeddings"):
    os.makedirs("embeddings")

if not os.path.isfile("embeddings/faiss_index"):
    vectors = FAISS.from_documents(docs, embeddings)
    vectors.save_local("embeddings/faiss_index")
    # persist_directory = 'db'
    # vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    # vectordb.persist()

print("Ingestion successful!!!")