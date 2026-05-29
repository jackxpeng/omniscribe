# OmniScribe: Privacy-First AI Process Manager

OmniScribe is a production-grade, privacy-first **Retrieval-Augmented Generation (RAG) Process Manager** designed for senior engineering teams. It parses and ingests raw meeting transcripts, vectorizes and stores them semantically in a PostgreSQL database using `pgvector`, extracts concrete action items with structured schemas, and runs an autonomous, non-blocking worker thread that simulates ticket execution sweeps (e.g., calling Jira or GitHub APIs).

Furthermore, OmniScribe integrates a comprehensive **Test-Driven Development (TDD) Evaluation Framework** using `promptfoo` to assert and validate output quality (using LLM-as-a-judge), latency, structure, and cost metrics locally and securely.

---

## 🚀 Key Architectural Pillars

*   **Native Async Concurrency Stack:** Built using FastAPI and `psycopg` (Psycopg 3) in fully asynchronous mode. Reusable warm TCP database sockets are managed via `AsyncConnectionPool`, unblocking the ASGI event loop completely during heavy I/O operations.
*   **Vector Search & Database Migration:** Utilizes the PostgreSQL `pgvector` extension. Our database is configured with high-fidelity `VECTOR(3072)` columns that map to Google's state-of-the-art `gemini-embedding-001` model.
*   **Dual-Layered TDD Evaluations:** Combines isolated prompt validation with full End-to-End (E2E) Integration Testing. Promptfoo uses a custom Python provider (`promptfoo_provider.py`) to verify the entire RAG retrieval and completing pipeline.
*   **Real-time Slot Observability:** Features a background server-sent events (SSE) stream (`/stream_slots`) that dynamically polls and visualizes local LLM slot VRAM telemetry and token decoding progress. *(Note: Gracefully defaults to "Engine offline" in cloud-only mode, shifting performance matrix/visual tracing to promptfoo).*

---

## 🛠️ System Architecture

Refer to [architecture_documentation.md](architecture_documentation.md) in the project root for complete, in-depth breakdowns featuring:
*   **C4 Level 1 (System Context)** and **C4 Level 2 (Container Architecture)** diagrams.
*   **UML Sequence Diagrams** mapping out parallel telemetry streaming and extraction loops.
*   **Event Loop Lifecycle Diagrams** outlining Python's cooperative multitasking scheduling.
*   **LLM-as-a-Judge Sequence Charts** mapping out promptfoo's semantic grading logic.

---

## 📦 Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your system:
*   Python 3.13 or newer
*   Node.js (v20.6.0+ or newer) and npm
*   PostgreSQL running locally (with the `pgvector` extension installed)

### 2. Configure Environment Variables
Create a `.env` file in the root of the project next to `pyproject.toml` and add your Gemini API key:
```env
GEMINI_API_KEY=your-gemini-api-key-here
```

### 3. Sync Dependencies
Sync your virtual environment with `uv` (our primary virtual env manager):
```bash
uv sync
```

---

## 🏃 Running the Application

### 1. Start the FastAPI Server
Launch the asynchronous FastAPI application on port `8001`:
```bash
uv run python main.py
```
*Expected Logs:*
```text
INFO:     Started server process [666622]
INFO:     Waiting for application startup.
Database initialized successfully.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```
*(The lifespan manager automatically runs startup migrations, registers `pgvector` async adapters on all pooled connections, and ensures schemas exist).*

### 2. Seed the Vector Database (The Golden Dataset)
With the server running on port `8001`, execute the database seeding script to populate PostgreSQL with our rich architecture transcript:
```bash
uv run python seed_db.py
```

### 3. Browse the Frontend Dashboard
Open your browser and navigate to:
```text
http://localhost:8001/
```
From here you can:
*   **Extract Actions:** Input an architectural query to vectorize, execute database similarity searches, and prompt Gemini to extract structured JSON tasks.
*   **Observe Telemetry:** Watch LLM token decoding progress in real-time.
*   **Execute Pending Tasks:** Trigger the Process Manager Sweep to process pending tasks, simulate API latency asynchronously, and persist `COMPLETED` states to PostgreSQL.

---

## 🧪 Test-Driven Development (TDD) & Evaluations

We implement a dual-layered evaluation architecture using **`promptfoo`** to ensure output quality (Quality), response times (Latency), and correct syntax (Structure).

### 1. Running E2E Integration Tests (6-Query Golden Dataset)
Our E2E suite tests the entire RAG pipeline (FastAPI routing $\rightarrow$ PostgreSQL pgvector similarity search $\rightarrow$ Gemini 2.5 Flash reasoning $\rightarrow$ Javascript validation). 

With your backend server running, execute promptfoo:
```bash
npx promptfoo@latest eval --no-cache
```
*Expected Output:*
```text
✓ Eval complete (ID: eval-xGx-2026-05-28T19:37:38)
Results:
  ✓ 6 passed (100%)
  0 failed (0%)
  0 errors (0%)
Duration: 12s (concurrency: 4)
```

### 2. Launching promptfoo Observability View
To visually inspect prompts, trace exact input/output tokens, and review why a specific probabilistic assertion passed or failed:
```bash
npx promptfoo@latest view
```
Open `http://localhost:15500` to browse promptfoo's built-in **Observability UI Dashboard**.

---

## 🎛️ Local vs. Cloud Observability Shift

Because OmniScribe supports both **local-first** execution (via a local model engine like `llama.cpp` or `Ollama`) and **cloud-native** pipelines (using the Google Gemini API), the observability tools adapt dynamically to your current execution mode:

### 1. Local-First Mode (VRAM & Slot Telemetry)
*   **Active Server:** Requires running a local llama.cpp/Ollama server at `http://localhost:11434`.
*   **Behavior:** The FastAPI server spins up a background generator `fetch_slot_data` that continuously polls local engine VRAM allocation and decoding speeds, streaming them via SSE (`/stream_slots`) to the frontend's real-time telemetry card.

### 2. Cloud-Native Mode (Cloud API & Phoenix/Promptfoo Observability)
*   **Active Server:** Using `gemini-2.5-flash` and `gemini-embedding-001` via API keys in your `.env`.
*   **Behavior:** Since cloud endpoints don't expose local hardware telemetry, the `/stream_slots` endpoint gracefully yields an `Engine offline` status to the frontend.
*   **New Observability Focus:** Visual tracing and runtime performance tracking shift entirely to **Arize Phoenix** and **Promptfoo**:
    *   **Arize Phoenix (Port 6006):** Runs a local trace collector automatically when you start the FastAPI server. Phoenix traces the entire RAG pipeline hierarchy in real-time, visualizing exact execution latencies and nested spans:
        1. `postgresql_vector_search` (Custom OTel Span tracking database similarity search times)
        2. `gemini-embedding-001` (Auto-instrumented OpenAI embedding call)
        3. `gemini-2.5-flash` (Auto-instrumented OpenAI chat completion call capturing raw inputs, tokens, and temperature)
    *   **Promptfoo (Port 15500):** Visualizes the offline evaluation matrix and charts semantic quality (LLM-as-a-judge scoring rubrics) across the Golden Dataset.

