# OmniScribe: System Architecture & Technical Design

This document serves as a complete architectural reference and learning guide for the OmniScribe Process Manager. It covers the system's design patterns, database lifecycles, non-blocking sequence flows, and the core engineering decisions made to transition the codebase from a hybrid thread-pool model to a production-grade fully asynchronous architecture.

---

## 1. System Context (C4 Level 1)

This diagram shows how OmniScribe sits within the engineering workspace, interacting with the end-user and local microservices.

```mermaid
graph TD
    User([Software Engineer])
    
    subgraph Local Workspace
        App[OmniScribe System]
        DB[(PostgreSQL + pgvector)]
    end

    subgraph Cloud Services
        Gemini[Google Gemini API]
    end

    User -->|Ingests transcripts, extracts & sweeps tasks| App
    App -->|Saves context & queries similarity| DB
    App -->|Generates embeddings & performs reasoning| Gemini
```

---

## 2. Container Architecture (C4 Level 2)

This diagram maps the internal components of OmniScribe, focusing on ports, runtimes, and protocols.

```mermaid
graph TB
    subgraph Client Browser [Client Browser]
        UI[Vanilla HTML5 / JS UI]
    end

    subgraph FastAPI Application Container [FastAPI Backend]
        API[Uvicorn ASGI Server: Port 8001]
        SSE[SSE Stream Generator]
        Pool[Psycopg 3 AsyncConnectionPool]
    end

    subgraph Persistence Layer [Database Container]
        DB[(Postgres Port 5432)]
    end

    subgraph Cloud API [Google Gemini Cloud API]
        GeminiAPI[Gemini API Endpoints]
    end

    subgraph Optional Local Legacy [Optional Local Inference]
        LocalEngine[Local llama.cpp Server: Port 11434]
    end

    UI -->|REST POST /ingest, /extract_actions, /execute_tasks| API
    UI -->|EventSource SSE GET /stream_slots| SSE
    API -->|Async HTTP Client /embeddings| GeminiAPI
    API -->|Async HTTP Client /chat/completions| GeminiAPI
    SSE -.->|Optional telemetry polling| LocalEngine
    Pool -->|Non-blocking sockets| DB
    API -->|Acquires connections| Pool
```

---

## 3. Core Architectural Sequences (UML)

### A. Non-Blocking Actions Extraction Flow (`/extract_actions`)
This sequence demonstrates how a user query traverses vector search, LLM completion, DB persistence, and frontend rendering, all while the parallel `/stream_slots` endpoint monitors LLM slots.

```mermaid
sequenceDiagram
    autonumber
    actor User as Engineer
    participant UI as Browser UI
    participant API as FastAPI Backend
    participant Pool as Async Connection Pool
    participant Gemini as Google Gemini API
    participant DB as PostgreSQL

    User->>UI: Click "Extract Actions"
    par Start Telemetry Channel (Graceful Default)
        UI->>API: GET /stream_slots (SSE Connection)
        loop Every 500ms
            API->>API: Poll local server (None found)
            API-->>UI: Push Event ("error": "Engine offline")
        end
    and Start Execution Channel
        UI->>API: POST /extract_actions (payload)
        activate API
        
        API->>Gemini: POST /embeddings (query text)
        Gemini-->>API: Vector array (float[3072])
        
        API->>Pool: Acquire Connection
        Pool->>API: AsyncConnection
        
        API->>DB: Cosine Distance query (<=> vector)
        DB-->>API: Text Context blocks
        
        API->>Gemini: POST /chat/completions (context + query)
        activate Gemini
        Note over Gemini: Cloud-based token decoding
        Gemini-->>API: Structured JSON response
        deactivate Gemini
        
        API->>DB: INSERT into action_items (PENDING)
        DB-->>API: Insert Success
        
        API-->>UI: HTTP 200 (Action items JSON list)
        deactivate API
    end
```

---

## 4. Key Architectural Decisions (ADR)

### Decision 1: Fully Asynchronous (`async/await`) over Synchronous Thread Pools
* **Context:** In early iterations, blocking database calls (`psycopg2`) and LLM requests were executed inside synchronous `def` endpoints. While FastAPI offloads `def` endpoints to a background thread pool, thread pools introduce overhead, do not scale to massive concurrency, and are susceptible to **thread starvation** under heavy loads.
* **Decision:** We migrated the entire system to native `async def` handlers.
* **Result:** The backend is now fully cooperative. While waiting for database sockets or LLM HTTP responses, the main thread yields control back to the ASGI event loop. This ensures that the SSE telemetry generator `/stream_slots` is guaranteed immediate CPU scheduling and remains perfectly responsive, even during intense background database write spikes.

### Decision 2: Psycopg 3 with Connection Pooling & Auto-Configuration
* **Context:** Open and close database handshakes represent significant TCP overhead (often 10–50ms of socket latency). Standard async drivers like `asyncpg` require custom wrappers to register vector adapters on every newly opened connection.
* **Decision:** We implemented Psycopg 3's `AsyncConnectionPool` and leveraged its native async configuration hook:
  ```python
  async def configure_conn(conn: psycopg.AsyncConnection):
      await register_vector_async(conn)
  ```
* **Result:** Warm database sockets are maintained and reused instantly. The `configure` parameter acts as a database connection interceptor, guaranteeing that every single database connection pulled from the pool is pre-registered to understand vector embeddings.

### Decision 3: Transition from Local hardware Slot Telemetry to Cloud APIs & Promptfoo Benchmarking
* **Context:** The application was initially designed around local inference engines (like `llama.cpp` or `Ollama`) tracking slot concurrency (`/stream_slots`). However, local LLMs introduce extreme latency, high local computing overhead, and complicate CI/CD pipelines.
* **Decision:** We migrated our core RAG model to **Google Gemini API** endpoints (using standard OpenAI-compatible structures). To handle this shift without UI breakage, the local `/stream_slots` endpoint gracefully handles connection failures by returning `{"error": "Engine offline"}`. In its place, the core observability system was shifted to **Promptfoo Test-Driven Evaluations** that track cost, precise cloud token limits, and latency directly in a dashboard.
* **Result:** Drastic improvement in query execution speeds (RAG extraction completed under 2 seconds) and standardized E2E quality validations via robust cloud endpoints.

---

## 5. The Event Loop Lifecycle & Cooperative Multitasking

To fully master this architecture, it is essential to understand Python's event loop behavior during database operations and simulated sweeps.

```mermaid
flowchart TD
    subgraph Event Loop [Main Thread Event Loop]
        Idle([Loop Idle]) --> |Check Event Queue| Active[Schedule Coroutine]
        Active --> Ingest[POST /ingest]
        Active --> Extract[POST /extract_actions]
        Active --> Telemetry[GET /stream_slots]
        Active --> Sweep[POST /execute_tasks]
    end

    Ingest -->|await get_embedding| Yield1[Yield Thread Control]
    Extract -->|await reasoning_client| Yield2[Yield Thread Control]
    Sweep -->|await asyncio.sleep| Yield3[Yield Thread Control]
    Telemetry -->|await asyncio.sleep| Yield4[Yield Thread Control]

    Yield1 --> Idle
    Yield2 --> Idle
    Yield3 --> Idle
    Yield4 --> Idle
```

### The Rules of Cooperative Multitasking:
1. **Never block the event loop:** Calling synchronous blocking methods (e.g., standard `time.sleep()`, synchronous `requests.get()`, or synchronous SQL queries) freezes the loop. No other task can execute.
2. **Always await I/O operations:** By using `await`, a task explicitly signals: *"I am waiting on a socket; suspend my execution and run other tasks on this thread."*
3. **CPU-bound work remains single-threaded:** In Python, the GIL restricts bytecode execution to one thread. However, because LLM inference and PostgreSQL lookups are network/socket operations (I/O-bound), the GIL is released during awaits, achieving massive concurrent performance.

---

## 6. Test-Driven Development (TDD) for Probabilistic Outputs

In deterministic software, a unit test asserts `actual == expected`. However, large language model outputs are **probabilistic** (non-deterministic). To apply TDD to LLM outputs, we employ a multi-layered evaluation framework that measures:
1. **Quality:** Captured through semantic checks and **"LLM-as-a-judge"** assertions.
2. **Structure:** Ensuring output matches syntax requirements (e.g., JSON schema).
3. **Performance (Latency & Cost):** Validating response times and token overhead.

### LLM-as-a-Judge Architecture & Sequence (promptfoo)

The following diagram illustrates how the `promptfoo` testing suite runs evaluations against our target model (`gemini-2.5-flash`) and grades results using a secondary "judge" model:

```mermaid
sequenceDiagram
    autonumber
    participant Suite as Promptfoo Test Runner
    participant Model as Gemini 2.5 Flash (Target LLM)
    participant Judge as Gemini 2.5 Flash (LLM Judge)

    Suite->>Model: 1. Send Prompt & Variables (Context, Query)
    Model-->>Suite: 2. Return Probabilistic Output (Raw JSON string)
    
    par Structural Check
        Suite->>Suite: 3. Verify is-json (Pass/Fail)
    and Performance Checks
        Suite->>Suite: 4. Verify Latency <= 12,000ms
        Suite->>Suite: 5. Verify Token Usage
    and Semantic Quality Check (LLM-as-a-judge)
        Suite->>Judge: 6. Send Rubric Criteria & Model Output
        Note over Judge: Grades output based on semantic rubric
        Judge-->>Suite: 7. Return Evaluation Verdict (Pass/Fail + Reasoning)
    end
    Suite->>Suite: 8. Consolidate Metrics & Log to local View Server
```

### Key Technical Takeaways for Probabilistic TDD:
* **The LLM Grader Rubric:** Model-graded assertions (like `llm-rubric`) provide semantic checks that deterministic regexes cannot catch. For instance, asserting that the assignee and task fields match meeting turns semantically, regardless of word choices.
* **Early Observability:** Storing evaluation runs locally allows you to open `promptfoo view` (`http://localhost:15500`) to visually inspect prompt runs, trace exact token counts, and review step-by-step judge reasoning to debug why a model failed a test.

