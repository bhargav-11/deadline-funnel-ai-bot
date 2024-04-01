from langchain_community.document_loaders.sitemap import SitemapLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import csv
# Load environment variables from .env file
load_dotenv()

OPENAI_API_TOKEN = os.getenv('OPENAI_API_KEY')

print("start")
text_splitter = RecursiveCharacterTextSplitter()
sitemap_loader = SitemapLoader(web_path="site.xml", is_local=True)
print("loading")
docs = sitemap_loader.load_and_split(text_splitter)
print("loaded")
print(docs)

if not os.path.isdir("embeddings"):
    os.makedirs("embeddings")

# Create a CSV file and open it for writing
with open('chunks.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # Write each chunk as a row in the CSV file
    for doc in docs:
        print(doc)
        writer.writerow([doc.page_content, doc.metadata])

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