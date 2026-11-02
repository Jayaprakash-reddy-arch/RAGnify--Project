from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss # Used to know the embedding length (vector length)
from langchain_community.vectorstores import FAISS # For creating Database



#STEP 1 : LOAD PDF

pdf_path = "2025_AnnualReport.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()
# STEP 2 : SPLIT DOCUMENTS INTO CHUNKS:-

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# loading chunks

chunks = text_splitter.split_documents(documents)

print("Number of documents:", len(documents))
print("Number of chunks:", len(chunks))


# STEP 3 : GENERATE EMBEDDINGS:-
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# STEP 4 : CREATE FAISS VECTOR STORE

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)


# STEP 5: SAVE FAISS DATABASE

vectorstore.save_local("faiss_db")

print("FAISS database saved successfully!")