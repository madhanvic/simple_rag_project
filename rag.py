import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load Environment variables from .env file
load_dotenv()


# Load PDF
print("Loading pdf...............")
loader = PyPDFLoader("pdf/TechCorp_Official_Employee_Handbook.pdf")
document = loader.load()
print(document[0].page_content)



# Chunk Text
print("Chunking text...............")
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = text_spliter.split_documents(document)
print(chunks[0].page_content)


# Vector database
print("Creating Vector database......................")
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_db = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings,
    persist_directory= "./chroma_db"
)


# Configure the database to act as a retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# Define the hidden prompt structure for the llm
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template=template)


# Initialize the ollama LLM
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

user_question = "What is the working hours and convert them in to IST?"
print(f"\nQuestion: {user_question}")

response = rag_chain.invoke(user_question)


# Clean up the output
if isinstance(response.content, list):
    clean_answer = response.content[0]['text']
else:
    clean_answer = response.content

print(f"Answer: {clean_answer}")