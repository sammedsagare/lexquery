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
    Detect article number either at the start (preferred) or mentioned anywhere in the text.
    Returns something like: 'Article 21'
    """
    patterns = [
        r"(?i)\barticle\s*(\d{1,3}[A-Z]?)\.?\s*-?\s*",  
        r"\b(\d{1,3}[A-Z]?)\.\s+",                     
        r"(?i)\bart\.?\s*(\d{1,3}[A-Z]?)",             
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"Article {match.group(1)}"
        
    fallback_pattern = r"(?i)\barticle\s+(\d{1,3}[A-Z]?)"
    match = re.search(fallback_pattern, text)
    if match:
        return f"Article {match.group(1)} (mentioned)"
    
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
            chunk.metadata["article_heading"] = heading
            chunk.page_content = f"{heading}\n{chunk.page_content}"

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_save_path)

    return vectorstore

def get_similar_chunks(query: str, k: int = 5):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = vectorstore.similarity_search(query, k=k)
    
    context = " ".join([doc.page_content for doc in docs])
    main_article = extract_article_heading(query)

    sources = [doc.metadata.get("article_heading", "Unknown Section") for doc in docs]
    sources = list(set(sources))
    if "Unknown Section" in sources:
        sources.remove("Unknown Section")
    if main_article and main_article in sources:
        sources.remove(main_article)
        sources.insert(0, main_article)

    sources = sources[:2]  # for now i am only showing the top 2 sources, ordered by priority

    return context, sources

def get_response_from_query(query: str, context: str) -> str:
    prompt = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
You are a helpful assistant answering questions about the Constitution of India.

Use ONLY the context below to answer the question.

Context:
{docs}

Question: {question}

If the context does not contain the exact answer, but provides relevant hints (example: phrases like "protection of life and personal liberty"), explain those as the basis of the answer but explain in depth and after all this if the context still does not contain any relevant information, respond with "I don't know".
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
    # vector_db_from_pdf(str(pdf_file))
    if not Path("faiss_index").exists():
        vector_db_from_pdf(str(pdf_file))
    
    
    query = input("Enter your query: ")

    print(f"\nWorking on it...\n")
    context, sources = get_similar_chunks(query)

    
    print("\nGenerating answer...\n")
    answer = get_response_from_query(query, context)

    print("\nAnswer:")
    print(answer)
    print("\nSource(s):")
    for i, src in enumerate(sources, 1):
        print(f"{i}. {src}")