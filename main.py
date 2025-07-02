from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def vector_db_from_pdf(pdf_path: str, index_save_path: str = "faiss_index") -> FAISS:
    
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s) from: {pdf_path}")    
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        
        chunk_overlap=200,     
        separators=["\n\n", "\n", ".", ";", ",", " "],  
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.\n")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index created and saved.")
    
    return vectorstore

if __name__ == "__main__":
    pdf_file = Path("constitution_of_india.pdf")
    vector_db_from_pdf(str(pdf_file))