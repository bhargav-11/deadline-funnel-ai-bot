
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
# Load the .env file to get the OpenAI API key
load_dotenv()
OPENAI_API_TOKEN = os.getenv('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_TOKEN, model="text-embedding-3-large")
llm = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.4, api_key=OPENAI_API_TOKEN)

llm_3_5 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

index = FAISS.load_local("embeddings/faiss_index", embeddings, allow_dangerous_deserialization=True)

# persist_directory = 'db'
# index = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# docstore_dict = index._collection.__dict__
# print(docstore_dict)


template = """
You are a helpful customer service bot for deadline funnel.
You need to support customer inquiries with best of your abilities using below provided context.
You are directly chatting with the customers.
The customers are very new to the product and their knowledge about the product is limited so give as much detailed answers as possible.
if the provided question is outside of the context, polietly respond you dont know.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Do not prompt users to reach out to email or phone.

Context: {context}

Customer's Question: {question}

Chat History: {chat_history}

Answer: 
"""

PROMPT = PromptTemplate(template=template, input_variables=['context', 'question', 'chat_history'])
retriever=index.as_retriever(k=5)

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=compression_retriever, combine_docs_chain_kwargs={"prompt": PROMPT}, return_source_documents=True
)

# print("hi-----------")
# print(compression_retriever.get_relevant_documents("eqweqweqweq"))
# print(len(compression_retriever.get_relevant_documents("eqweqweqweq")))

Human_message = """
Prompt: Given a question, identify and extract the key search term relevant for conducting a search. Focus on identifying main term that are essential for finding specific information related to the query. 
Keep your answers concise.

Question: {question}

Search Term:

"""

while True:
    chat_history = []
    question = input("Enter your question (type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    response = chain.invoke({"question": question, "chat_history": chat_history})

    answer = response.get('answer')

    chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    print("\n")
    print(f"{Colors.YELLOW}{answer}{Colors.RESET}")
    print("\n")

    # Print the most accurate 2 sources
    source_documents = response.get('source_documents')
    if source_documents:
        for i, document in enumerate(source_documents[:5]):
            print(f"{Colors.CYAN}Source {i+1}: {document.metadata['source']}{Colors.RESET}")
    
    # Get the search term
    search_term_messages = [
        SystemMessage(
            content="You are a helpful assistant to extract the key search term"
        ),
        HumanMessage(
            content=Human_message.format(question=question)
        ),
    ]
    search_term = llm_3_5.invoke(search_term_messages)
    print("\n")
    print(f"{Colors.RED}Search Term: {search_term.content}{Colors.RESET}")