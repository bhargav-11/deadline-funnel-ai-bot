from langchain_community.document_loaders.sitemap import SitemapLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores.chroma import Chroma

OPENAI_API_TOKEN = os.getenv('OPENAI_API_KEY')

web_links = [
    "https://www.growthleap.com/blog/how-to-create-an-evergreen-webinar-funnel-15/",
    "https://www.growthleap.com/blog/how-scarcity-works-and-why-its-digital-marketings-best-friend/",
    "https://www.growthleap.com/blog/how-to-create-an-evergreen-email-funnel/",
    "https://www.growthleap.com/blog/how-to-build-your-evergreen-launch-funnel-step-by-step/",
    "https://www.growthleap.com/blog/5-tips-to-maximize-sales-from-your-evergreen-marketing-funnel/",
    "https://www.growthleap.com/blog/how-to-avoid-launch-burnout-with-an-evergreen-funnel-system/",
    "https://www.growthleap.com/blog/ultimate-black-friday-guide-for-online-course-creators/",
    "https://www.growthleap.com/blog/how-to-create-an-evergreen-virtual-summit/",
    "https://www.growthleap.com/blog/2022-evergreen-summit/"
]

print("start")
texts = []
for link in web_links:
    text_splitter = RecursiveCharacterTextSplitter()
    loader = WebBaseLoader(link)
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

# Save the updated index
index.save_local("embeddings/faiss_index")

print("Ingestion successful!!!")