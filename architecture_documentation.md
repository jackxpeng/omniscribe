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
        Nomic[Nomic Embeddings API]
        Gemma[Gemma 4 Reasoning API]
    end

    User -->|Ingests transcripts, extracts & sweeps tasks| App
    App -->|Saves context & queries similarity| DB
    App -->|Generates text vectors| Nomic
    App -->|Performs LLM inference| Gemma
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

    subgraph LLM Inference Containers [Local Inference Infrastructure]
        NomicEmbed[Nomic Microservice: Port 11435]
        GemmaEngine[Gemma 4 LLM Server: Port 11434]
    end

    UI -->|REST POST /ingest, /extract_actions, /execute_tasks| API
    UI -->|EventSource SSE GET /stream_slots| SSE
    API -->|Async HTTP Client /v1/embeddings| NomicEmbed
    API -->|Async HTTP Client /v1/chat/completions| GemmaEngine
    SSE -->|Async HTTP GET /slots| GemmaEngine
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
    participant Nomic as Nomic Server (11435)
    participant Gemma as Gemma Server (11434)
    participant DB as PostgreSQL

    User->>UI: Click "Extract Actions"
    par Start Telemetry Channel
        UI->>API: GET /stream_slots (SSE Connection)
        loop Every 500ms
            API->>Gemma: GET /slots
            Gemma-->>API: Array of slot states (Slot 1 processing = true)
            API-->>UI: Push Event ("tokens_decoded": 285)
        end
    and Start Execution Channel
        UI->>API: POST /extract_actions (payload)
        activate API
        
        API->>Nomic: POST /v1/embeddings (query text)
        Nomic-->>API: Vector array (float[768])
        
        API->>Pool: Acquire Connection
        Pool->>API: AsyncConnection
        
        API->>DB: Cosine Distance query (<=> vector)
        DB-->>API: Text Context blocks
        
        API->>Gemma: POST /v1/chat/completions (context + query)
        activate Gemma
        Note over Gemma: Processes prompt & decodes tokens
        Gemma-->>API: Structured JSON response
        deactivate Gemma
        
        API->>DB: INSERT into action_items (PENDING)
        DB-->>API: Insert Success
        
        API-->>UI: HTTP 200 (Action items JSON list)
        deactivate API
    end
    UI->>UI: Close SSE Connection
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

### Decision 3: Dynamically Searching Slot Telemetry
* **Context:** Local LLM engines like `llama.cpp` process multiple prompts concurrently using separate slots. Statically reading the first slot (`data[0]`) in the front-end resulted in false "Idle" readouts whenever the engine assigned prompt execution to Slot 1, 2, or 3.
* **Decision:** Replaced static array reading with a dynamic predicate search:
  ```javascript
  const activeSlot = data.find(slot => slot.is_processing === true) || data[0];
  ```
* **Result:** The dashboard maintains perfect observability across multi-user, multi-slot environments.

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
