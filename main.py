from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

pdf_path = Path("aies.pdf")
loader = PyMuPDFLoader(str(pdf_path))
documents = loader.load()

print(f"✅ Loaded {len(documents)} page(s) from: {pdf_path.name}")
