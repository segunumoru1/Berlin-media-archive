# 🏛️ Berlin Media Archive - Multi-Modal RAG System

A production-grade Multi-Modal Retrieval-Augmented Generation (RAG) system for searching across historical audio and document archives with precise source attribution.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Assessment Requirements](#assessment-requirements)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Production Considerations](#production-considerations)
- [Project Structure](#project-structure)

---

## 🎯 Overview

The Berlin Media Archive is designed to solve a real-world problem: enabling historians and journalists to search across 50+ years of archived content including radio interviews, panel discussions, and manuscripts in both German and English.

**Example Query:**
> "What did Christa Wolf say about reunification in her 1990 interview?"

**Expected Response:**
> Christa Wolf expressed both hope and anxiety about reunification, stating "Reunification brings hope, but also anxiety. We must not forget the experiences of those who lived through the division" (Source: interview_1990.mp3, 04:23). She emphasized concerns about rapid change in her writings (Source: history_book.pdf, Page 45).

---

## ✨ Features

### Core Requirements (Part 1) ✅

- **🎙️ Audio Ingestion Pipeline**
  - Transcription using OpenAI Whisper
  - Timestamp preservation for every segment
  - Support for multiple audio formats
  - Metadata extraction

- **📄 Unified Vector Store**
  - ChromaDB for vector storage
  - Rich metadata tracking (source, type, timestamps, page numbers)
  - Efficient chunking with overlap
  - OpenAI embeddings (text-embedding-3-large)

- **📊 Attribution Engine**
  - Precise source citations (timestamps for audio, page numbers for documents)
  - No hallucinations - all claims must be cited
  - Natural language answers with inline citations

### Advanced Modules (Part 2) ✅

- **🔍 Module A: Hybrid Search**
  - Combines semantic search (vector similarity) with keyword search (BM25)
  - Handles specific dates, names, and proper nouns
  - Configurable weighting between semantic and keyword results

- **👥 Module B: Speaker Diarization**
  - Automatic speaker identification using pyannote.audio
  - Speaker-specific filtering in queries
  - Timeline tracking for each speaker

- **📈 Module C: Evaluation System**
  - LLM-as-Judge evaluation with GPT-4
  - Metrics: Faithfulness, Relevance, Citation Quality
  - RAGAS integration for additional metrics
  - Automated quality scoring (0-1 scale, A-F grades)

### Production Standards (Part 3) ✅

- **🛡️ Error Handling**
  - Graceful degradation on service failures
  - Comprehensive error messages
  - Retry logic with exponential backoff

- **📝 Observability**
  - Structured logging with loguru
  - Request tracing
  - Performance metrics
  - Query analytics

- **🧪 Testing**
  - Unit tests for core components
  - Integration tests for pipelines
  - Evaluation test cases
  - Coverage reporting with pytest-cov

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Interface                      │
│                  (API / CLI / Web Interface)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Attribution Engine (RAG Core)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Query Processing & Hybrid Search                 │  │
│  │     • Semantic Search (OpenAI Embeddings)            │  │
│  │     • Keyword Search (BM25)                          │  │
│  │     • Metadata Filtering                             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  2. Context Retrieval                                │  │
│  │     • Top-K Selection                                │  │
│  │     • Re-ranking                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  3. Answer Generation (GPT-4)                        │  │
│  │     • Citation-aware prompting                       │  │
│  │     • Source attribution                             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vector Database (ChromaDB)                 │
│  ┌────────────────────┐    ┌─────────────────────────────┐ │
│  │  Audio Chunks      │    │  Document Chunks            │ │
│  │  • Transcript text │    │  • PDF text                 │ │
│  │  • Timestamps      │    │  • Page numbers             │ │
│  │  • Speaker IDs     │    │  • Section metadata         │ │
│  │  • Embeddings      │    │  • Embeddings               │ │
│  └────────────────────┘    └─────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Ingestion Pipelines                        │
│  ┌────────────────────┐    ┌─────────────────────────────┐ │
│  │  Audio Pipeline    │    │  Document Pipeline          │ │
│  │  • Whisper API     │    │  • PyPDF                    │ │
│  │  • Diarization     │    │  • Chunking                 │ │
│  │  • Embedding       │    │  • Embedding                │ │
│  └────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- (Optional) HuggingFace token for speaker diarization
- FFmpeg for audio processing

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd berlin-media-archive
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional (for speaker diarization)
HUGGINGFACE_TOKEN=your-huggingface-token-here

# Configuration
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-large
WHISPER_MODEL=base
ENABLE_SPEAKER_DIARIZATION=true
ENABLE_HYBRID_SEARCH=true
```

### Step 5: Install FFmpeg (for audio processing)

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

---

## 🎬 Quick Start

### Option 1: Interactive Demo

```bash
python demo.py
```

This launches an interactive demo with:
- Audio ingestion examples
- Document processing examples
- Query demonstrations
- Evaluation system showcase

### Option 2: CLI Tool

```bash
# Check system status
python cli.py status

# Ingest audio file
python cli.py ingest audio path/to/audio.mp3

# Ingest PDF document
python cli.py ingest document path/to/document.pdf

# Query the archive
python cli.py query "What is discussed about success?"

# Run evaluation
python cli.py evaluate run
```

### Option 3: API Server

```bash
# Start API server
python main.py

# Or using CLI
python cli.py server
```

Then visit:
- API Documentation: http://localhost:8000/docs
- Interactive UI: http://localhost:8000

---

## 📖 Usage

### 1. Audio Ingestion

```python
from ingestion.audio_ingestion import AudioIngestionPipeline

# Initialize pipeline
pipeline = AudioIngestionPipeline(
    whisper_model="base",
    enable_diarization=True
)

# Process audio file
segments, metadata = pipeline.ingest_audio(
    audio_path="interview_1990.mp3",
    output_dir="./output/audio"
)

# Access transcript segments
for segment in segments:
    print(f"[{segment.get_timestamp_str()}] {segment.speaker}: {segment.text}")
```

**Output:**
```
[00:15 - 00:23] SPEAKER_01: Welcome to our discussion...
[00:24 - 00:35] SPEAKER_02: Thank you for having me...
```

### 2. Document Ingestion

```python
from ingestion.document_ingestion import DocumentIngestionPipeline

# Initialize pipeline
pipeline = DocumentIngestionPipeline(
    chunk_size=1000,
    chunk_overlap=200
)

# Process document
chunks, metadata = pipeline.ingest_document(
    document_path="history_book.pdf",
    output_dir="./output/documents"
)

# Access chunks
for chunk in chunks:
    print(f"Page {chunk.page_number}: {chunk.text[:100]}...")
```

### 3. Querying the Archive

```python
from rag.attribution_engine import AttributionEngine

# Initialize engine
engine = AttributionEngine()

# Query with natural language
result = engine.query(
    question="What did Christa Wolf say about reunification?",
    top_k=5,
    filters={"source_type": "audio"}  # Optional: filter by audio only
)

# Display results
print(f"Answer: {result['answer']}")
print(f"\nSources:")
for source in result['sources']:
    print(f"  - {source['citation']}")
```

**Output:**
```
Answer: Christa Wolf expressed concerns about rapid reunification, 
stating "We must not forget the experiences" (Source: interview_1990.mp3, 04:23)...

Sources:
  - interview_1990.mp3, 04:23 (SPEAKER_02)
  - history_book.pdf, Page 45
```

### 4. Evaluation

```python
from rag_evaluator.llm_rag_evaluator import LLMRAGEvaluator
from rag_evaluator.test_cases import EvaluationTestCases

# Initialize evaluator
evaluator = LLMRAGEvaluator(model="gpt-4-turbo-preview")

# Get test cases
test_cases = EvaluationTestCases.get_sample_test_cases()

# Run batch evaluation
results = evaluator.batch_evaluate(
    test_cases,
    save_results=True,
    output_path="./output/evaluation_results.json"
)

# View results
print(f"Average Score: {results.avg_overall:.2f}")
print(f"Grade Distribution: {results.grade_distribution}")
print(f"Production Ready: {results.production_ready_percentage:.1f}%")
```

---

## 📊 Assessment Requirements

### ✅ Part 1: Core Requirements (MVP)

| Requirement | Status | Implementation |
|------------|--------|----------------|
| Audio ingestion with timestamps | ✅ Complete | [`ingestion/audio_ingestion.py`](ingestion/audio_ingestion.py) |
| Unified vector store with metadata | ✅ Complete | [`vectorstore/`](vectorstore/) |
| Attribution engine with citations | ✅ Complete | [`rag/attribution_engine.py`](rag/attribution_engine.py) |

### ✅ Part 2: Advanced Modules

| Module | Status | Implementation |
|--------|--------|----------------|
| Module A: Hybrid Search | ✅ Complete | [`rag/hybrid_search.py`](rag/hybrid_search.py) |
| Module B: Speaker Diarization | ✅ Complete | [`ingestion/audio_ingestion.py`](ingestion/audio_ingestion.py) |
| Module C: Evaluation Metrics | ✅ Complete | [`rag_evaluator/`](rag_evaluator/) |

### ✅ Part 3: Production Standards

| Standard | Status | Implementation |
|----------|--------|----------------|
| Error handling | ✅ Complete | Throughout codebase |
| Observability | ✅ Complete | [`utils/logger.py`](utils/logger.py) |
| Testing | ✅ Complete | [`tests/`](tests/) |

### ✅ Part 4: System Design

See [`DESIGN.md`](DESIGN.md) for detailed scalability analysis, cost estimates, and video expansion strategy.

---

## 🔌 API Documentation

### Endpoints

#### Health Check
```http
GET /api/v1/health
```

#### Ingest Audio
```http
POST /api/v1/ingest/audio
Content-Type: multipart/form-data

file: <audio_file>
enable_diarization: true
```

#### Ingest Document
```http
POST /api/v1/ingest/document
Content-Type: multipart/form-data

file: <pdf_file>
```

#### Query Archive
```http
POST /api/v1/query
Content-Type: application/json

{
  "question": "What is discussed about success?",
  "top_k": 5,
  "filters": {
    "source_type": "audio",
    "speaker": "SPEAKER_02"
  }
}
```

**Response:**
```json
{
  "answer": "Success is defined as...",
  "sources": [
    {
      "citation": "interview_1990.mp3, 04:23",
      "type": "audio",
      "text": "...",
      "metadata": {
        "speaker": "SPEAKER_02",
        "timestamp": "04:23"
      }
    }
  ],
  "confidence": 0.95
}
```

#### Evaluate Response
```http
POST /api/v1/evaluate
Content-Type: application/json

{
  "question": "What is success?",
  "answer": "Success is achieving goals...",
  "context": ["Context 1...", "Context 2..."],
  "ground_truth": "Success means..."
}
```

For full API documentation, visit `/docs` when running the server.

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_audio_ingestion.py -v
```

### Test Structure

```
tests/
├── test_audio_ingestion.py      # Audio pipeline tests
├── test_document_ingestion.py   # Document pipeline tests
├── test_attribution_engine.py   # RAG system tests
├── test_hybrid_search.py        # Search tests
├── test_evaluation.py           # Evaluation tests
└── test_integration.py          # End-to-end tests
```

---

## 🏭 Production Considerations

### Performance

- **Batch Processing**: Process multiple files concurrently
- **Caching**: Cache embeddings and frequently accessed data
- **Connection Pooling**: Reuse database connections
- **Rate Limiting**: Respect API rate limits

### Scalability

See [`DESIGN.md`](DESIGN.md) for detailed scalability architecture including:
- Horizontal scaling strategies
- Database sharding approaches
- Caching layers
- Load balancing

### Monitoring

```python
# Enable detailed logging
LOG_LEVEL=DEBUG

# Monitor key metrics
- Query latency
- Embedding generation time
- Vector search performance
- API response times
```

### Security

- API key rotation
- Input validation and sanitization
- Rate limiting per user
- Audit logging for sensitive queries

---

## 📁 Project Structure

```
berlin-media-archive/
├── api/                          # FastAPI routes
│   └── router.py                 # API endpoints
├── data/                         # Data storage
│   ├── audio/                    # Audio files
│   ├── documents/                # PDF documents
│   └── vectorstore/              # ChromaDB data
├── embeddings/                   # Embedding services
│   ├── __init__.py
│   └── openai_embeddings.py      # OpenAI embeddings
├── ingestion/                    # Data ingestion
│   ├── __init__.py
│   ├── audio_ingestion.py        # Audio pipeline
│   └── document_ingestion.py     # Document pipeline
├── rag/                          # RAG system
│   ├── __init__.py
│   ├── attribution_engine.py     # Main RAG engine
│   └── hybrid_search.py          # Hybrid search
├── rag_evaluator/                # Evaluation system
│   ├── __init__.py
│   ├── llm_rag_evaluator.py      # LLM-as-Judge
│   ├── ragas_evaluator.py        # RAGAS integration
│   ├── metrics.py                # Evaluation metrics
│   └── test_cases.py             # Test cases
├── tests/                        # Test suite
│   └── ...
├── utils/                        # Utilities
│   ├── config.py                 # Configuration
│   └── logger.py                 # Logging setup
├── vectorstore/                  # Vector DB
│   └── chroma_store.py           # ChromaDB wrapper
├── .env                          # Environment variables
├── cli.py                        # Command-line interface
├── demo.py                       # Interactive demo
├── main.py                       # API server
├── requirements.txt              # Dependencies
├── DESIGN.md                     # System design doc
└── README.md                     # This file
```

---

## 🎓 Example Queries

### Query 1: Specific Date
```
"What was said in 1989?"
```
→ Uses hybrid search to find exact date mentions

### Query 2: Speaker-Specific
```
"What did the guest say about success?"
```
→ Filters by speaker metadata

### Query 3: Cross-Modal
```
"Compare the interview's discussion of reunification with the book's analysis"
```
→ Retrieves from both audio and document sources

### Query 4: Temporal
```
"What changed between the 1990 and 2023 interviews?"
```
→ Uses timestamp filtering and comparison

---

## 🤝 Contributing

This is an assessment project. For production use, please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 and Whisper API
- **pyannote.audio** for speaker diarization
- **ChromaDB** for vector storage
- **RAGAS** for evaluation framework
- **LangChain** for RAG utilities

---

## 📞 Support

For questions or issues:
1. Check the [documentation](#api-documentation)
2. Run the [demo](#option-1-interactive-demo)
3. Review [test cases](tests/)
4. Check logs in `logs/` directory

---

## 🎯 Assessment Completion Checklist

- [x] Audio ingestion with timestamps
- [x] Document ingestion with page tracking
- [x] Unified vector store
- [x] Attribution engine with citations
- [x] Hybrid search (Module A)
- [x] Speaker diarization (Module B)
- [x] Evaluation system (Module C)
- [x] Error handling
- [x] Logging and observability
- [x] Unit tests
- [x] API server
- [x] CLI tool
- [x] Interactive demo
- [x] Comprehensive documentation
- [x] System design document
- [x] Cost analysis
- [x] Video expansion strategy

---

**Built with ❤️ for the Berlin Media Archive Technical Assessment**