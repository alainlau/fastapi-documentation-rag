# FastAPI Documentation RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) assistant built for the **AAI 655O: Bridging Search and Generative AI with RAG** university course.

The assistant answers questions about FastAPI using a selected collection of official FastAPI documentation. It retrieves relevant documentation passages, generates grounded answers with source citations, and refuses questions when the retrieved evidence is insufficient.

## Features

- 20 selected FastAPI documentation pages covering request validation, dependencies, security, error handling, and database integration.
- Structure-aware Markdown parsing and chunking.
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Persistent ChromaDB vector storage with cosine similarity.
- Similarity retrieval with configurable parameters.
- LLM-based answer generation with numbered source citations.
- Support for Gemini, OpenAI, and Ollama.

## Requirements

- Python 3.11 or newer, with the version in `.python-version` preferred.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environment and dependency management.
- An API key for Gemini or OpenAI, or a locally running Ollama server.

**Gemini is the default provider used for the project's evaluation.**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/alainlau/fastapi-documentation-rag.git
cd fastapi-documentation-rag
```

### 2. Install dependencies

```bash
uv sync
```

This creates the project's virtual environment and installs its dependencies.

The first run may download the local embedding model from Hugging Face.

### 3. Configure the language model

Copy `.env.example` to `.env`.

On Windows:

```powershell
copy .env.example .env
```

On Linux or macOS:

```bash
cp .env.example .env
```

For Gemini, configure the relevant values in `.env`:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

Alternatively, configure `LLM_PROVIDER=openai` with `OPENAI_API_KEY`, or `LLM_PROVIDER=ollama` with a running local Ollama server.

Do not commit your `.env` file or API keys.

## Running the project

### Step 1: Prepare the documentation corpus

```bash
uv run python prepare_corpus.py
```

This selects the 20 documentation pages used by the project from the included FastAPI documentation files.

### Step 2: Build the vector database

```bash
uv run python ingest.py
```

The ingestion process parses the Markdown documentation, expands referenced Python examples, splits the content into chunks, generates embeddings, and stores them in ChromaDB.

The evaluated configuration produced **555 indexed chunks**.

The database is saved locally in `chroma_db/`.

### Step 3: Start the assistant

```bash
uv run python app.py
```

Once ready, the assistant displays:

```text
FastAPI RAG Assistant ready!
Type 'exit' to quit.
```

Enter a question and press Enter. The assistant prints the retrieved evidence, including source paths and similarity scores, followed by the generated answer.

Type `exit` to close the application.

## Example questions

Try the following questions:

1. How do I validate incoming JSON data in FastAPI?
2. How can I make an endpoint accessible only to logged-in users?
3. How can I return a custom error when a request contains invalid data?
4. How do I reuse the same database connection logic across several endpoints?
5. What will the weather be tomorrow?

The final question is intentionally outside the documentation's scope and demonstrates the assistant's refusal behavior.

The assistant may also refuse in-scope questions when its retrieved evidence is insufficient.

## Retrieval configuration

The main retrieval settings are defined in `config.py`:

| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Distance metric | Cosine |
| Retrieval method | Similarity |
| Top-k | 12 |
| Similarity threshold | 0.35 |
| Maximum embedding input | 240 tokens |

The assistant uses the same embedding model for indexing and querying.

## Project files

| File / directory | Purpose |
|---|---|
| `app.py` | Interactive question-answering application |
| `ingest.py` | Embedding generation and ChromaDB indexing |
| `config.py` | Central configuration |
| `prepare_corpus.py` | Selects the documentation corpus |
| `markdown_parser.py` | Parses Markdown and expands Python examples |
| `compare_chunking.py` | Compares chunking configurations |
| `inspect_corpus.py` | Inspects corpus statistics |
| `test_parser.py` | Tests documentation parsing |
| `data/` | Included FastAPI documentation and selected corpus |
| `chroma_db/` | Locally generated vector database |

## Notes

- The assistant is an academic prototype, not a production application.
- Answers are generated using retrieved documentation rather than unrestricted model knowledge.
- Source citations such as `[1]` refer to the numbered evidence passages printed before each answer.
- If the available documentation does not sufficiently support an answer, the assistant responds:

  "I cannot answer this question based on the provided documentation."

- The ChromaDB index can
