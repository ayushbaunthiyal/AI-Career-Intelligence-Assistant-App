# 🎯 Career Intelligence Assistant

A RAG-based conversational AI assistant that analyzes resumes against job descriptions to help candidates understand skill gaps, experience alignment, and prepare for interviews.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Architecture Overview](#-architecture-overview)
- [RAG/LLM Approach & Decisions](#-ragllm-approach--decisions)
- [Key Technical Decisions](#-key-technical-decisions)
- [Engineering Standards](#-engineering-standards)
- [Productionization & Scaling](#-productionization--scaling)
- [AI Tools in Development](#-ai-tools-in-development)
- [Future Improvements](#-future-improvements)
- [Known Limitations](#-known-limitations)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Docker Desktop (optional, for containerized deployment)

### Local Development Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd AI-Career-Intelligence-Assistant-App

# 2. Install uv (if not already installed)
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv sync

# 4. Configure environment variables
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run the application
uv run streamlit run app/main.py
```

The app will be available at `http://localhost:8502`

### Docker Deployment

```bash
# 1. Configure environment
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Build and run with Docker Compose
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The app will be available at `http://localhost:8502`

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html
```

---

## ✨ Features

- **Resume Upload**: Upload your resume (PDF/DOCX) - new uploads replace existing
- **Multiple Job Postings**: Add multiple job descriptions to compare against
- **Skill Gap Analysis**: "What skills am I missing for Job #1?"
- **Experience Alignment**: "How does my experience align with the requirements?"
- **Interview Preparation**: "Help me prepare for the interview"
- **Multi-Job Comparison**: "Which role am I best suited for?"
- **Conversational Memory**: Multi-turn conversations with context awareness

---

## 📸 Screenshots

### Document Upload Interface
The upload section supports both file uploads (PDF/DOCX) and text paste for job postings.

![Upload Interface](docs/screenshots/upload_interface.png)
*Upload your resume and multiple job postings via file upload or text paste*

### Document Management Sidebar
The sidebar shows all uploaded documents with management options and statistics.

![Sidebar](docs/screenshots/sidebar_documents.png)
*View uploaded documents, manage your library, and see statistics*

### Career Fit Analysis
The AI provides detailed analysis with skill match percentages and fit recommendations.

![Analysis Results](docs/screenshots/analysis_results.png)
*Get detailed analysis with skill match percentages and fit recommendations*

### Chat Interface
Interactive chat interface with streaming responses and source citations.

![Chat Interface](docs/screenshots/chat_interface.png)
*Ask questions and get intelligent responses with source citations*

> **Note**: Screenshot files should be placed in `docs/screenshots/` directory. The images above are placeholders - replace them with actual screenshots of your running application.

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   Upload    │  │  Chat Interface │  │  Document Sidebar   │  │
│  └─────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Service Layer                             │
│  ┌─────────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ DocumentProcessor│  │ TextChunker │  │    RAGService       │  │
│  │ (PDF/DOCX parse) │  │ (Semantic)  │  │ (LangChain Chain)   │  │
│  └─────────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Storage & AI Layer                        │
│  ┌─────────────────────────┐      ┌───────────────────────────┐ │
│  │       ChromaDB          │      │       OpenAI API          │ │
│  │  (Vector Store - Local) │      │  • text-embedding-3-small │ │
│  │                         │      │  • gpt-4o-mini            │ │
│  └─────────────────────────┘      └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Upload**: User uploads resume/job posting
2. **Processing**: DocumentProcessor extracts text from PDF/DOCX
3. **Chunking**: TextChunker splits into semantic chunks (512 tokens, 50 overlap)
4. **Embedding**: OpenAI creates embeddings for each chunk
5. **Storage**: ChromaDB stores chunks with metadata
6. **Query**: User asks a question
7. **Retrieval**: ChromaDB finds relevant chunks via MMR search
8. **Generation**: LLM generates response using retrieved context
9. **Display**: Streamlit shows response with source citations

---

## 🧠 RAG/LLM Approach & Decisions

### Choices Considered

| Component | Options Considered | Final Choice | Rationale |
|-----------|-------------------|--------------|-----------|
| **LLM** | GPT-4o, GPT-4o-mini, Claude | **gpt-4o-mini** | Best cost/quality balance for career advice; sufficient reasoning capability |
| **Embeddings** | text-embedding-3-small, text-embedding-3-large, local (Sentence Transformers) | **text-embedding-3-small** | 1536 dims, excellent quality, cost-effective ($0.02/1M tokens) |
| **Vector DB** | ChromaDB, Pinecone, Qdrant, FAISS | **ChromaDB** | Local/free, easy setup, LangChain integration, metadata filtering |
| **Orchestration** | LangChain, LlamaIndex, Custom | **LangChain** | Mature ecosystem, good docs, conversational chain support |
| **Chunking** | Fixed-size, Semantic, Document-structure | **RecursiveCharacterTextSplitter** | Respects semantic boundaries (paragraphs > sentences > words) |

### Chunking Strategy

```python
# 512 tokens (~2000 chars) with 50 token overlap
# Optimized for resume sections and job posting bullet points
separators = ["\n\n", "\n", ". ", ", ", " ", ""]
```

**Why these settings:**
- 512 tokens captures full resume sections (Education, Experience entries)
- 50 token overlap ensures context isn't lost at boundaries
- Semantic separators preserve meaning (paragraph > line > sentence)

### Retrieval Approach

- **Method**: Maximum Marginal Relevance (MMR)
- **Top-K**: 5 documents
- **Why MMR**: Provides diverse results, avoiding redundant chunks from same section

### Prompt Engineering

**System Prompt Design:**
- Defined clear persona: "Career Intelligence Assistant"
- Explicit capabilities and guidelines
- Guardrails to keep responses career-focused
- Instructions for handling job references (#1, #2, etc.)

**Context Management:**
- Dynamic context assembly from retrieved chunks
- Chat history limited to last 5 turns (prevent token bloat)
- Source document metadata preserved for citations

### Guardrails & Quality Controls

1. **Input Validation**: File type/size checks before processing
2. **Output Guardrails**: System prompt constrains to career topics
3. **Error Handling**: Graceful degradation with user-friendly messages
4. **Source Citations**: Every response shows source documents used

### Observability

- **Logging**: Structured logging throughout services
- **LangSmith Ready**: Can enable LangSmith for chain tracing (optional)
- **Metrics**: Document stats visible in sidebar

---

## 🔧 Key Technical Decisions

### 1. Single Resume Model

**Decision**: Users can only have one resume (new upload replaces old)

**Rationale**:
- Matches the assignment's example queries ("my skills", "my experience")
- Simpler UX - no ambiguity about which resume
- Most users compare ONE resume against multiple jobs
- Easy to extend later if multi-resume needed

### 2. ChromaDB for Local Development

**Decision**: Use ChromaDB with local persistence

**Rationale**:
- Zero cost (assignment requirement)
- No external dependencies or API keys needed
- Persistent across app restarts
- Easy migration path to cloud (Pinecone, Qdrant Cloud)

### 3. Streaming Responses

**Decision**: Stream LLM responses token-by-token

**Rationale**:
- Better UX - users see response forming
- Perceived performance improvement
- Standard in modern chat interfaces

### 4. Session-Based Services

**Decision**: Initialize services once per Streamlit session

**Rationale**:
- Avoid reinitializing ChromaDB on every interaction
- Maintain conversation memory across turns
- Better performance

### 5. Metadata-Rich Chunks

**Decision**: Store extensive metadata with each chunk

```python
metadata = {
    "doc_type": "resume" | "job_posting",
    "filename": "original_file.pdf",
    "chunk_index": 0,
    "total_chunks": 5,
    "word_count": 150,
}
```

**Rationale**:
- Enables filtering by document type
- Supports job-specific queries ("Job #1")
- Allows source attribution in responses

---

## 📐 Engineering Standards

### Code Organization

```
app/
├── main.py              # Application entry point
├── config.py            # Pydantic settings (12-factor app)
├── services/            # Business logic (single responsibility)
├── components/          # UI components (separation of concerns)
├── prompts/             # Prompt templates (maintainability)
└── utils/               # Helper functions (DRY)
```

### Patterns Applied

- **Dependency Injection**: Services receive dependencies, not create them
- **Configuration as Code**: Pydantic Settings for type-safe config
- **Separation of Concerns**: UI, business logic, storage are distinct
- **Single Responsibility**: Each module has one clear purpose

### Type Safety

- Full type hints throughout codebase
- Pydantic for data validation
- Dataclasses for domain objects

### Testing Strategy

- **Unit Tests**: Core services (document processing, chunking, helpers)
- **Fixtures**: Shared test data and mocks in `conftest.py`
- **Isolation**: Mocked external dependencies (OpenAI, settings)

### Code Quality Tools

```toml
# pyproject.toml
[tool.ruff]         # Linting & formatting
[tool.mypy]         # Type checking
[tool.pytest]       # Testing
```

---

## 🚀 Productionization & Scaling

### Current State → Production Ready

| Area | Current (MVP) | Production Requirement |
|------|---------------|----------------------|
| **Vector DB** | ChromaDB (local) | Pinecone / Qdrant Cloud / Weaviate |
| **Authentication** | None | OAuth2 / JWT + API keys |
| **Storage** | Local filesystem | S3 / GCS for documents |
| **Caching** | None | Redis for embeddings cache |
| **Monitoring** | Basic logging | Prometheus + Grafana + LangSmith |
| **Deployment** | Docker Compose | Kubernetes / ECS / Cloud Run |

### AWS Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Route 53 (DNS)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Application Load Balancer                     │
│                    (SSL termination, health checks)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ECS Fargate                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Streamlit Container (auto-scaling based on CPU/memory) │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pinecone  │    │   S3 Bucket     │    │  Secrets Manager│
│ (Vector DB) │    │ (Documents)     │    │  (API Keys)     │
└─────────────┘    └─────────────────┘    └─────────────────┘
```

### Scaling Considerations

1. **Horizontal Scaling**: Stateless app design allows multiple instances
2. **Vector DB Scaling**: Pinecone auto-scales; Qdrant supports sharding
3. **Embedding Caching**: Cache embeddings to reduce OpenAI costs
4. **Rate Limiting**: Implement per-user rate limits for API calls
5. **Async Processing**: Queue document processing for large files

### Security Hardening

1. **API Key Rotation**: Use AWS Secrets Manager with rotation
2. **Input Sanitization**: Already implemented file type/size validation
3. **Network Isolation**: VPC with private subnets for services
4. **Audit Logging**: CloudTrail for compliance
5. **Data Encryption**: At-rest (S3 SSE) and in-transit (TLS)

---

## 🤖 AI Tools in Development

### g. How I Used AI Tools in Development

**Cursor AI (Claude)** was used for:

**Cursor AI (Claude)** was used for:

1. **Boilerplate Generation**: Initial project structure, config files, Docker setup
2. **Code Scaffolding**: Service class skeletons with proper typing
3. **Documentation**: Docstrings and inline comments
4. **Test Generation**: Unit test cases following established patterns

### My Development Workflow with AI

1. **Design First**: I planned architecture and made tech decisions BEFORE coding
2. **Iterative Refinement**: Generated code, reviewed, modified to my preferences
3. **Manual Review**: Every generated file was reviewed for:
   - Correctness
   - Consistency with project conventions
   - Error handling edge cases
4. **Testing**: Ran tests to verify AI-generated code works

### Do's and Don'ts with AI Coding Assistants

**Do:**
- ✅ Use for boilerplate and repetitive code
- ✅ Use for generating test cases from existing code
- ✅ Use for documentation drafts
- ✅ Always review and understand generated code
- ✅ Refactor AI output to match your coding style

**Don't:**
- ❌ Blindly accept generated code without review
- ❌ Use for complex business logic without understanding
- ❌ Skip testing because "AI generated it"
- ❌ Let AI make architectural decisions without your input
- ❌ Use AI output directly in documentation without your perspective

### Making AI-Assisted Code Maintainable

1. **Consistent Style**: Applied ruff formatting across all files
2. **Clear Abstractions**: Ensured AI-generated code follows project patterns
3. **Type Hints**: Verified all functions have proper typing
4. **Comments for "Why"**: Added explanations for non-obvious decisions

---

## 🔮 Future Improvements

*What I'd do with more time:*

### High Priority

1. **Better Resume Parsing**: Use dedicated resume parsing library (e.g., `pyresparser`) for structured extraction (skills, education, experience sections)

2. **Job Matching Score**: Quantitative fit score (0-100%) based on keyword matching + semantic similarity

3. **Streaming Sources**: Currently sources are fetched separately; integrate into streaming response

4. **Session Persistence**: Save chat history to allow resuming conversations

### Medium Priority

5. **Multi-Modal Support**: Parse resume images/screenshots using GPT-4 Vision

6. **Skill Ontology**: Use a skills taxonomy (ESCO, O*NET) for better skill matching

7. **Interview Question Bank**: Generate role-specific interview questions from job descriptions

8. **Export Functionality**: Export analysis as PDF report

### Nice to Have

9. **Resume Tailoring**: AI-powered suggestions to tailor resume for specific jobs

10. **Comparative Analytics**: Visualize skill gaps across multiple jobs

11. **Integration**: LinkedIn job import, ATS integration

---

## ⚠️ Known Limitations

1. **PDF Parsing Quality**: Complex PDF layouts (multi-column, tables) may not extract perfectly
2. **No OCR**: Scanned PDFs won't work (text extraction only)
3. **English Only**: Prompts and analysis optimized for English content
4. **Single Session**: Data doesn't persist across browser sessions (by design for MVP)
5. **No Authentication**: Anyone with the URL can access (fine for local dev)
6. **Rate Limits**: OpenAI rate limits may affect heavy usage

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- UI powered by [Streamlit](https://streamlit.io/)
- LLM by [OpenAI](https://openai.com/)
