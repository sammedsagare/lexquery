import os
import re
import json
import shutil
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

DEFAULT_PERSIST_DIR = Path("chroma_index")
DEFAULT_PDF_FILES = [
    Path("docs/constitution_of_india.pdf"),
    Path("docs/BNS.pdf"),
    Path("docs/BNSS.pdf"),
    Path("docs/BSA.pdf"),
]
SUPPORTED_LAWS = {
    "BNS": "Bharatiya Nyaya Sanhita",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
    "BSA": "Bharatiya Sakshya Adhiniyam",
}
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 500
MAX_CONTEXT_CHARS = 3000
CHUNKING_VERSION = 2
INDEX_METADATA_FILE = "index_metadata.json"

PROMPT = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
You are a legal assistant specializing in Indian law. You are helping the user across multiple questions in the same conversation.

Your goal is to provide accurate, step-by-step legal advice using only the information in the context below. Always consider the full conversation history, not just the current question.

Use relevant laws from:
- Constitution of India
- Bharatiya Nyaya Sanhita (BNS) - The new criminal code
- Bharatiya Nagarik Suraksha Sanhita (BNSS) - The new criminal procedure code
- Bharatiya Sakshya Adhiniyam (BSA) - The new evidence act

Respond in the following format:

Advisory:
- Provide a clear, step-by-step response addressing the user's current question.
- If this question builds on a previous one, carry over the relevant context or assumptions.
- Reference the current legal framework: BNS for criminal law, BNSS for criminal procedure, and BSA for evidence law.

Citations:
- List all relevant sections/articles clearly at the end.
- IMPORTANT: Prioritize sections from BNS (criminal law), BNSS (criminal procedure), and BSA (evidence law) as they are the current legal framework.
- Format examples:
  - Article 22(2) of the Constitution of India
  - Section 103 of the BNS
  - Section 41 of the BNSS
  - Section 25 of the BSA

Rules:
- Use ONLY the information from the context provided.
- PRIORITIZE citing sections from BNS, BNSS, and BSA as they are the current legal codes.
- When referencing criminal law, use BNS; for criminal procedure, use BNSS; for evidence law, use BSA.
- NEVER invent information outside the legal sources.
- If the documents do not contain enough info to answer accurately, respond with:
  "I don't know - the legal documents provided do not contain enough information to answer this question."

Context:
{docs}

Question:
{question}
"""
)

_ARTICLE_PATTERNS = [
    (re.compile(r"(?i)\barticle\s*(\d{1,3}[A-Z]?)\b"), "Article {}"),
    (re.compile(r"(?i)\bsection\s*(\d{1,3}[A-Z]?)\b"), "Section {}"),
    (re.compile(r"(?i)\bart\.?\s*(\d{1,3}[A-Z]?)\b"), "Article {}"),
    (re.compile(r"(?i)\bsec\.?\s*(\d{1,3}[A-Z]?)\b"), "Section {}"),
]
_STRUCTURE_PATTERNS = [
    re.compile(r"(?im)^(article|section)\s+[A-Z0-9().-]+\b.*$"),
    re.compile(r"(?im)^(chapter|part)\s+[A-Z0-9().-]+\b.*$"),
    re.compile(r"(?m)^\d{1,3}[A-Z]?\.\s+.+$"),
]

_EMBEDDINGS: HuggingFaceEmbeddings | None = None
_VECTORSTORE: Chroma | None = None
_VECTORSTORE_PATH: str | None = None
_LLM: ChatGroq | None = None

memory = InMemoryChatMessageHistory()


def get_embeddings() -> HuggingFaceEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2"
        )
    return _EMBEDDINGS


def get_llm() -> ChatGroq:
    global _LLM
    if _LLM is None:
        _LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    return _LLM


def extract_article_heading(text: str) -> str | None:
    for pattern, template in _ARTICLE_PATTERNS:
        match = pattern.search(text)
        if match:
            return template.format(match.group(1).upper())
    return None


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", ";", ":"],
    )


def validate_pdf_paths(pdf_paths: list[str | Path]) -> list[Path]:
    resolved_paths = [Path(path) for path in pdf_paths]
    missing_paths = [str(path) for path in resolved_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing PDF files: {', '.join(missing_paths)}")
    return resolved_paths


def normalize_legal_text(text: str) -> str:
    normalized_lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        normalized_lines.append(cleaned)
    text = "\n".join(normalized_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_id_for(text: str, metadata: dict) -> str:
    raw = "|".join(
        [
            str(metadata.get("law") or ""),
            str(metadata.get("start_page", "")),
            str(metadata.get("end_page", "")),
            str(metadata.get("article_heading") or ""),
            text[:200],
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def split_combined_text_into_sections(text: str) -> list[str]:
    starts = set()
    for pattern in _STRUCTURE_PATTERNS:
        for match in pattern.finditer(text):
            starts.add(match.start())

    ordered_starts = sorted(starts)
    if not ordered_starts:
        return [text]

    sections = []
    last_start = 0
    for start in ordered_starts:
        if start > last_start:
            candidate = text[last_start:start].strip()
            if candidate:
                sections.append(candidate)
        last_start = start

    tail = text[last_start:].strip()
    if tail:
        sections.append(tail)

    return sections


def extract_page_numbers(text: str) -> tuple[list[int], str]:
    page_numbers = [int(match.group(1)) for match in re.finditer(r"<<<PAGE:(\d+)>>>", text)]
    cleaned_text = re.sub(r"\n?<<<PAGE:\d+>>>\n?", "\n", text)
    return page_numbers, cleaned_text.strip()


def build_section_documents(pdf_path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(pdf_path))
    page_documents = loader.load()
    print(f"Loaded {len(page_documents)} page(s) from: {pdf_path}")

    page_blocks = []
    for page_doc in page_documents:
        page_number = int(page_doc.metadata.get("page", 0)) + 1
        page_text = normalize_legal_text(page_doc.page_content)
        if page_text:
            page_blocks.append(f"<<<PAGE:{page_number}>>>\n{page_text}")

    combined_text = "\n\n".join(page_blocks)
    raw_sections = split_combined_text_into_sections(combined_text)
    law_name = pdf_path.stem
    section_documents: list[Document] = []

    for section_text in raw_sections:
        page_numbers, cleaned_text = extract_page_numbers(section_text)
        if not cleaned_text:
            continue

        article_heading = extract_article_heading(cleaned_text)
        metadata = {
            "law": law_name,
            "source": law_name,
            "start_page": min(page_numbers) if page_numbers else None,
            "end_page": max(page_numbers) if page_numbers else None,
            "article_heading": article_heading,
            "chunk_type": "section",
        }

        prefixed_text = (
            f"{article_heading}\n{cleaned_text}" if article_heading and not cleaned_text.startswith(article_heading) else cleaned_text
        )
        metadata["chunk_id"] = chunk_id_for(prefixed_text, metadata)
        section_documents.append(Document(page_content=prefixed_text, metadata=metadata))

    return section_documents


def split_section_documents(section_documents: list[Document]) -> list[Document]:
    splitter = build_splitter()
    split_documents = splitter.split_documents(section_documents)

    for index, doc in enumerate(split_documents):
        article_heading = doc.metadata.get("article_heading")
        if article_heading and not doc.page_content.startswith(article_heading):
            doc.page_content = f"{article_heading}\n{doc.page_content}"
        doc.metadata["chunk_type"] = "subsection" if len(split_documents) > len(section_documents) else "section"
        doc.metadata["chunk_index"] = index
        doc.metadata["chunk_id"] = chunk_id_for(doc.page_content, doc.metadata)

    return split_documents


def get_index_metadata_path(persist_directory: str | Path) -> Path:
    return Path(persist_directory) / INDEX_METADATA_FILE


def build_index_metadata(pdf_paths: list[Path]) -> dict:
    return {
        "chunking_version": CHUNKING_VERSION,
        "files": [
            {
                "path": str(path),
                "name": path.name,
                "size": path.stat().st_size,
                "mtime": path.stat().st_mtime,
            }
            for path in pdf_paths
        ],
    }


def load_index_metadata(persist_directory: str | Path) -> dict | None:
    metadata_path = get_index_metadata_path(persist_directory)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def save_index_metadata(persist_directory: str | Path, metadata: dict) -> None:
    metadata_path = get_index_metadata_path(persist_directory)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def index_needs_rebuild(pdf_paths: list[Path], persist_directory: str | Path) -> bool:
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        return True
    stored_metadata = load_index_metadata(persist_path)
    current_metadata = build_index_metadata(pdf_paths)
    return stored_metadata != current_metadata


def reset_vectorstore_cache() -> None:
    global _VECTORSTORE, _VECTORSTORE_PATH
    _VECTORSTORE = None
    _VECTORSTORE_PATH = None


def rebuild_vector_db(
    pdf_paths: list[str | Path], persist_directory: str | Path = DEFAULT_PERSIST_DIR
) -> Chroma:
    persist_path = Path(persist_directory)
    resolved_pdf_paths = validate_pdf_paths(pdf_paths)

    if persist_path.exists():
        shutil.rmtree(persist_path)

    vectorstore = vector_db_from_pdfs(resolved_pdf_paths, persist_directory=persist_path)
    save_index_metadata(persist_path, build_index_metadata(resolved_pdf_paths))
    reset_vectorstore_cache()
    return vectorstore


def vector_db_from_pdfs(
    pdf_paths: list[str | Path], persist_directory: str | Path = DEFAULT_PERSIST_DIR
) -> Chroma:
    all_chunks = []

    for pdf_path in validate_pdf_paths(pdf_paths):
        section_documents = build_section_documents(pdf_path)
        chunks = split_section_documents(section_documents)
        all_chunks.extend(chunks)

    return Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
    )


def get_vectorstore(persist_directory: str | Path = DEFAULT_PERSIST_DIR) -> Chroma:
    global _VECTORSTORE, _VECTORSTORE_PATH
    persist_directory = str(persist_directory)
    if _VECTORSTORE is None or _VECTORSTORE_PATH != persist_directory:
        _VECTORSTORE = Chroma(
            persist_directory=persist_directory,
            embedding_function=get_embeddings(),
        )
        _VECTORSTORE_PATH = persist_directory
    return _VECTORSTORE


def ensure_vectorstore(
    pdf_paths: list[str | Path] | None = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
) -> Chroma:
    resolved_pdf_paths = validate_pdf_paths(pdf_paths or DEFAULT_PDF_FILES)
    persist_path = Path(persist_directory)
    if index_needs_rebuild(resolved_pdf_paths, persist_path):
        rebuild_vector_db(resolved_pdf_paths, persist_directory=persist_path)
    return get_vectorstore(persist_directory=persist_path)


def build_history_text(chat_memory: InMemoryChatMessageHistory) -> str:
    history_lines: list[str] = []
    for message in chat_memory.messages[-MAX_HISTORY_MESSAGES:]:
        role = "You" if isinstance(message, HumanMessage) else "Assistant"
        content = (
            message.content
            if isinstance(message.content, str)
            else " ".join(str(item) for item in message.content)
        )
        if len(content) > MAX_HISTORY_CHARS:
            content = f"{content[:MAX_HISTORY_CHARS]}... [truncated]"
        history_lines.append(f"{role}: {content.strip()}")
    return "\n".join(history_lines)


def build_full_context(context: str, chat_memory: InMemoryChatMessageHistory) -> str:
    trimmed_context = context
    if len(trimmed_context) > MAX_CONTEXT_CHARS:
        trimmed_context = (
            f"{trimmed_context[:MAX_CONTEXT_CHARS]}\n\n... [context truncated for token limit]"
        )

    history_text = build_history_text(chat_memory)
    return (
        f"--- Recent Conversation ---\n{history_text}\n\n"
        f"--- Retrieved Context ---\n{trimmed_context}"
    )


def get_similar_chunks(
    query: str, k: int = 12, persist_directory: str | Path = DEFAULT_PERSIST_DIR
) -> tuple[str, list[str]]:
    vectorstore = ensure_vectorstore(persist_directory=persist_directory)
    docs = vectorstore.similarity_search(query, k=k)

    law_sources: dict[str, list] = {}
    for doc in docs:
        law = doc.metadata.get("law", "unknown")
        law_sources.setdefault(law, []).append(doc)

    for law_code, law_name in SUPPORTED_LAWS.items():
        if law_code not in law_sources:
            extra_docs = vectorstore.similarity_search(f"{query} {law_code} {law_name}", k=2)
            docs.extend(
                doc for doc in extra_docs if doc.metadata.get("law", "").upper() == law_code
            )

    unique_docs = []
    seen_doc_ids = set()
    for doc in docs:
        doc_id = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:120])
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        unique_docs.append(doc)

    selected_docs = unique_docs[:k]
    context = "\n\n".join(doc.page_content for doc in selected_docs)

    sources = []
    for doc in selected_docs:
        article = doc.metadata.get("article_heading") or extract_article_heading(doc.page_content)
        law = doc.metadata.get("law", "Unknown Law").replace("_", " ").title()
        sources.append(f"{article or 'Unknown Section'} ({law})")

    return context, list(dict.fromkeys(sources))


def get_response_from_query(
    query: str, context: str, chat_memory: InMemoryChatMessageHistory | None = None
) -> str:
    active_memory = chat_memory or memory
    full_context = build_full_context(context, active_memory)
    chain = RunnableSequence(PROMPT, get_llm())
    response = chain.invoke({"question": query, "docs": full_context})

    answer = response.content.strip()
    active_memory.add_user_message(query)
    active_memory.add_ai_message(answer)
    return answer


def run_cli() -> None:
    ensure_vectorstore()

    while True:
        query = input("\nEnter your legal question (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            print("Please enter a question.")
            continue

        print("\nWorking on it...")
        context, sources = get_similar_chunks(query)
        print(f"Retrieved sources: {sources}")
        print("Generating answer...\n")
        answer = get_response_from_query(query, context, memory)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    run_cli()
