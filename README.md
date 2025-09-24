# LexQuery

This project is built to focus on Indian Law-related PDFs. It is an AI-powered legal assistant that answers questions using official legal documents (like the Constitution of India, BNS, BNSS, and BSA) in PDF format. It uses large language models (LLMs), semantic search, and conversation memory to provide accurate, context-aware, and citation-rich legal responses—strictly based on the content of your uploaded PDFs.

## Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/sammedsagare/lexquery.git
   cd talk_with_pdf
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate # Windows only
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set your environment variable for Groq API:

   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

1. Place your law PDFs (e.g., `constitution_of_india.pdf`, `BNS.pdf`, `BNSS.pdf`, `BSA.pdf`) in the project directory.
2. Run the main script:

   ```bash
   python main.py
   ```

3. On first run, the script will build a Chroma vector index from all provided PDFs. This may take a few minutes.
4. Enter your legal questions at the prompt. The assistant will use conversation memory to provide context-aware answers, citing relevant articles/sections from the loaded documents.
5. Type `exit` to quit the chat.

## Features

- Multi-document support: Constitution of India, BNS (Bharatiya Nyaya Sanhita), BNSS (Bharatiya Nagarik Suraksha Sanhita), BSA (Bharatiya Sakshya Adhiniyam) - India's current legal framework
- Fast semantic search using Chroma vector store
- Multi-turn conversation memory (remembers previous questions/answers)
- Legal citations with every answer
- No information is invented: answers are strictly based on the loaded documents

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies (notably: `langchain`, `langchain_chroma`, `langchain_huggingface`, `PyMuPDF`, `ChromaDB`, `groq`)

## Notes

- You must provide your own Groq API key for LLM responses.
- For best results, use high-quality, text-based PDFs.
- This system uses India's current legal framework: BNS (criminal law), BNSS (criminal procedure), and BSA (evidence law)
- The system prioritizes citations from the current legal codes over outdated references
- To add more laws, simply add their PDF filenames to the `pdf_files` list in `main.py`

## Relevant Links

- [ChatGroq (Groq Cloud LLMs)](https://console.groq.com/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Chroma Vector Store](https://docs.trychroma.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
