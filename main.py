from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import re

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")


def extract_article_heading(text: str) -> str | None:
    """
    Detects article numbers in formats like:
    - 21. Protection...
    - Article 370...
    - ART. 22A Right...
    """
    patterns = [
        r"(?i)\barticle\s*(\d{1,3}[A-Z]?)\.?\s*-?\s*", 
        r"\b(\d{1,3}[A-Z]?)\.\s+",                     
        r"(?i)\bART\.?\s*(\d{1,3}[A-Z]?)",             
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"Article {match.group(1)}"
    return None

def vector_db_from_pdf(pdf_path: str, index_save_path: str = "faiss_index") -> FAISS:
    
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s) from: {pdf_path}")    
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,        
        chunk_overlap=200,     
        separators=["\n\n", "\n", ".", ";", ",", " "],  
    )
    
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        heading = extract_article_heading(chunk.page_content)
        if heading:
            chunk.page_content = f"{heading}\n{chunk.page_content}"

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_save_path)

    return vectorstore

def get_similar_chunks(query: str, k: int = 5):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    
    vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    docs = vectorstore.similarity_search(query, k=k)
    context = " ".join([doc.page_content for doc in docs])
    
    
    return context

def get_response_from_query(query: str, context: str) -> str:
    prompt = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
You are a helpful assistant answering questions about the Constitution of India.

Use ONLY the context below to answer the question.

Context:
{docs}

Question: {question}

If the context does not contain the answer, say "I don't know".
Be specific and cite the article numbers or key phrases wherever possible.
"""
)

    llm = ChatGroq(
        model="gemma2-9b-it",  
        temperature=0.7
    )

    chain = RunnableSequence(prompt, llm)

    response = chain.invoke({"question": query, "docs": context})

    return response.content.strip()

if __name__ == "__main__":
    pdf_file = Path("constitution_of_india.pdf")
    vector_db_from_pdf(str(pdf_file))
    
    query = input("Enter your query: ")

    print(f"\nWorking on it...'\n")
    context = get_similar_chunks(query)

    print("\nGenerating answer...\n")
    answer = get_response_from_query(query, context)

    print("\nAnswer:")
    print(answer)