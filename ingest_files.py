from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores.chroma import Chroma

OPENAI_API_TOKEN = os.getenv('OPENAI_API_KEY')

print("start")
text_splitter = RecursiveCharacterTextSplitter()
texts = []
file_directory = "files/"
file_extension_mapping = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader
}

for file_name in os.listdir(file_directory):
    file_path = os.path.join(file_directory, file_name)
    file_extension = os.path.splitext(file_name)[1]
    
    if file_extension in file_extension_mapping:
        loader_class = file_extension_mapping[file_extension]
        if loader_class == TextLoader:
            loader = loader_class(file_path, encoding="utf-8")
        else:
            loader = loader_class(file_path)
        docs = loader.load_and_split(text_splitter)
        texts.extend(docs)

print("loaded")
print(texts)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_TOKEN, model="text-embedding-3-large")

# if not os.path.isdir("embeddings"):
#     os.makedirs("embeddings")

# if not os.path.isfile("embeddings/faiss_index"):
#     vectors = FAISS.from_documents(docs, embeddings)

#     vectors.save_local("embeddings/faiss_index")
# else:
print("Loading existing index")
# persist_directory = 'db'
# vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# vectordb.add_texts(texts, embeddings)
# vectordb.persist()

index = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)

# Add the file to the index
index.add_documents(texts)

for doc_id, doc in index.docstore.__dict__["_dict"].items():
    print(doc.metadata)

# Save the updated index
index.save_local("embeddings/faiss_index")

print("Ingestion successful!!!")