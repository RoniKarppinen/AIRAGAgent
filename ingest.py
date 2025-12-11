import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Configurations
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "data_archive", "billgates.pdf")
persist_directory = os.path.join(current_dir, "chroma_db")
collection_name = "bill_gates_docs"

def ingest_documents():
    """Ingests PDF documents, splits them into chunks, creates embeddings, and stores them in a Chroma vector store."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    if os.path.exists(persist_directory):
        print(f"Clearing existing vector store at {persist_directory}...")
        shutil.rmtree(persist_directory)

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(pages)
    print(f"Created {len(documents)} text chunks.")

    print("Creating vector store (Embedding chunks)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print("--- Ingestion Complete ---")
    print(f"Database saved to: {persist_directory}")

if __name__ == "__main__":
    ingest_documents()