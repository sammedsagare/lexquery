from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import re
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

memory = InMemoryChatMessageHistory()


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


def get_response_from_query(query: str, context: str, memory: InMemoryChatMessageHistory) -> str:
    history_text = ""
    for msg in memory.messages:
        role = "You" if isinstance(msg, HumanMessage) else "Assistant"
        content_str = msg.content if isinstance(msg.content, str) else " ".join(str(item) for item in msg.content)
        history_text += f"{role}: {content_str.strip()}\n"

    full_context = f"--- Conversation so far ---\n{history_text}\n\n--- Retrieved context ---\n{context}"

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
You are a legal assistant specializing in Indian law. You are helping the user across multiple questions in the same conversation.

Your goal is to provide accurate, step-by-step legal advice using only the information in the context below. Always consider the full conversation history, not just the current question.

Use relevant laws from:
- Constitution of India
- Indian Penal Code (IPC)
- Code of Criminal Procedure (CrPC)

Respond in the following format:

Advisory:
• Provide a clear, step-by-step response addressing the user's current question.
• If this question builds on a previous one, carry over the relevant context or assumptions.

Citations:
• List all relevant sections/articles clearly at the end.
• Format examples:
  - Article 22(2) of the Constitution of India  
  - Section 41 of the CrPC  
  - Section 420 of the IPC

Rules:
- Use ONLY the information from the context provided.
- NEVER invent information outside the legal sources.
- If the documents do not contain enough info to answer accurately, respond with:  
  "I don't know — the legal documents provided do not contain enough information to answer this question."

Context:
{docs}

Question:
{question}

"""
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    chain = RunnableSequence(prompt, llm)

    response = chain.invoke({"question": query, "docs": full_context})

    memory.add_user_message(query)
    memory.add_ai_message(response.content.strip())

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
        answer = get_response_from_query(query, context, memory)

        print("\nAnswer:\n", answer)
