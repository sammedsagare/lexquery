from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import re

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")


def extract_article_heading(text: str) -> str | None:
    """
    Detect article / section number either at the start or mentioned anywhere in the text.
    Returns something like: 'Article 21' / 'Section 154'
    If no article/section is found, returns None.
    """
    patterns = [
        r"(?i)\barticle\s*(\d{1,3}[A-Z]?)\.?\s*-?\s*",
        r"\bsection\s*(\d{1,3}[A-Z]?)\.?\s*-?\s*",
        r"\b(\d{1,3}[A-Z]?)\.\s+",
        r"(?i)\bart\.?\s*(\d{1,3}[A-Z]?)",
        r"(?i)\bsec\.?\s*(\d{1,3}[A-Z]?)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return f"{match.group(0).strip().title()}"
    return None

def vector_db_from_pdfs(pdf_paths: list[str], persist_directory: str = "chroma_index") -> Chroma:
    all_chunks = []
    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"Loaded {len(documents)} page(s) from: {pdf_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ";", ",", " "],
        )
        chunks = splitter.split_documents(documents)
        law_name = Path(pdf_path).stem

        for chunk in chunks:
            heading = extract_article_heading(chunk.page_content)
            if heading:
                chunk.metadata["article_heading"] = heading
                chunk.page_content = f"{heading}\n{chunk.page_content}"
            chunk.metadata["law"] = law_name
            chunk.metadata["source"] = Path(pdf_path).stem

        all_chunks.extend(chunks)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore


def get_similar_chunks(query: str, k: int = 5):
    vectorstore = Chroma(
        persist_directory="chroma_index",
        embedding_function=embeddings
    )
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    sources = []
        
    for doc in docs:
        article = doc.metadata.get("article_heading")
        if not article:
            match = re.search(r"(?i)\b(article\s+\d{1,3}[A-Z]?)", doc.page_content)
            article = match.group(1).title() if match else "Unknown Section"

        law = doc.metadata.get("law", "Unknown Law").replace("_", " ").title()
        sources.append(f"{article} ({law})")

    sources = list(dict.fromkeys(sources))
    return context, sources


def get_response_from_query(query: str, context: str) -> str:
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
You are a legal assistant helping users based on Indian law.

Use ONLY the context below to answer the question. Look through all laws — Constitution of India, IPC, CrPC — and list any relevant articles or sections.

Context:
{docs}

Question: {question}

Your answer must follow this format:
1. Start with a clear, step-by-step advisory response.
2. At the end of your answer, ALWAYS, i repeat, ALWAYS cite the relevant section/article with law name (e.g., Section 41 of CrPC, Article 22(2) of Constitution).
3. Be accurate. If context lacks enough info, say "I don't know" with reasoning.

Be specific and legally sound. Do not guess.
"""
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    chain = RunnableSequence(prompt, llm)
    response = chain.invoke({"question": query, "docs": context})
    return response.content.strip()

if __name__ == "__main__":
    pdf_files = [
        "constitution_of_india.pdf",
        "IPC.pdf",
        "CrPC.pdf"
    ]
    persist_dir = "chroma_index"

    if not Path(persist_dir).exists():
        vector_db_from_pdfs(pdf_files, persist_directory=persist_dir)

    while True:
        query = input("\nEnter your legal question (or 'exit'): ")
        if query.lower() == "exit":
            break
        print("\nWorking on it...")
        context, sources = get_similar_chunks(query)
        print("Generating answer...\n")
        answer = get_response_from_query(query, context)

        print("\nAnswer:\n", answer)
        # print("\nSource(s):")
        # for i, src in enumerate(sources, 1):
        #     print(f"{i}. {src}")