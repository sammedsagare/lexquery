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

def get_similar_chunks(query: str, k: int = 5):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    print("📦 Loaded FAISS index from disk.")

    docs = vectorstore.similarity_search(query, k=k)
    context = " ".join([doc.page_content for doc in docs])
    
    print(f"🔍 Retrieved {len(docs)} relevant chunks for query: '{query}'\n")
    
    return context

if __name__ == "__main__":
    pdf_file = Path("constitution_of_india.pdf")
    vector_db_from_pdf(str(pdf_file))
    
    query = input("Enter your query: ")
    print(f"\n🔍 Searching for relevant chunks for query: '{query}'\n")
    context = get_similar_chunks(query)
    print(context[:1000])