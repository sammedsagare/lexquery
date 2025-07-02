from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = Path("constitution_of_india.pdf")
loader = PyMuPDFLoader(str(pdf_path))
documents = loader.load()

print(f"✅ Loaded {len(documents)} page(s) from: {pdf_path.name}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)

chunks = splitter.split_documents(documents)
print(f"✅ Split into {len(chunks)} chunks.\n")
