import json
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
import psycopg
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
from utils import parse_transcript_to_turns, partition_turns_to_parents

db_conninfo = "dbname=postgres user=postgres password=postgres host=localhost port=5432"

async def configure_conn(conn: psycopg.AsyncConnection):
    await register_vector_async(conn)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the asynchronous connection pool
    app.state.db_pool = AsyncConnectionPool(conninfo=db_conninfo, open=False, configure=configure_conn)
    await app.state.db_pool.open()
    
    # Initialize database schemas
    try:
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # Migrate to decoupled parent-child tables
                await cur.execute("DROP TABLE IF EXISTS child_chunks CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS parent_documents CASCADE;")
                await cur.execute("DROP TABLE IF EXISTS meeting_memory CASCADE;")
                
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS parent_documents (
                        id SERIAL PRIMARY KEY,
                        topic VARCHAR(255),
                        content TEXT
                    );
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS child_chunks (
                        id SERIAL PRIMARY KEY,
                        parent_id INTEGER REFERENCES parent_documents(id) ON DELETE CASCADE,
                        content TEXT,
                        embedding VECTOR(3072)
                    );
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS action_items (
                        id SERIAL PRIMARY KEY,
                        assignee VARCHAR(100),
                        task TEXT,
                        status VARCHAR(50) DEFAULT 'PENDING'
                    );
                """)
            print("Database initialized successfully.")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        
    yield
    
    # Clean shutdown of connection pool
    await app.state.db_pool.close()

app = FastAPI(title="OmniScribe Process Manager", lifespan=lifespan)

# 1. Initialize the Google Gemini OpenAI-compatible client
embedding_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)

reasoning_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY,
)


class TranscriptPayload(BaseModel):
    topic: str
    transcript: str


async def get_embedding(text: str) -> list[float]:
    """Calls Gemini embeddings API to convert text to a vector."""
    response = await embedding_client.embeddings.create(
        input=text,
        model="gemini-embedding-001",
    )
    return response.data[0].embedding


@app.post("/ingest")
async def ingest_transcript(payload: TranscriptPayload):
    try:
        # Step 1: Parse and partition transcript into Parent-Child relationships
        turns = parse_transcript_to_turns(payload.transcript)
        if not turns:
            raise HTTPException(
                status_code=400, detail="Transcript could not be parsed into speaker turns."
            )

        # Group turns into parent blocks (up to 1500 chars), tracking which turns belong to each parent
        parent_blocks = partition_turns_to_parents(turns, max_chars=1500)

        # Step 2: Insert parent documents and embed/insert individual child chunks (speaker turns)
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                for block in parent_blocks:
                    # 2a. Insert parent document and get ID
                    await cur.execute(
                        "INSERT INTO parent_documents (topic, content) VALUES (%s, %s) RETURNING id",
                        (payload.topic, block["content"]),
                    )
                    parent_id = (await cur.fetchone())[0]

                    # 2b. Insert each speaker turn inside the block as a child chunk
                    for turn in block["turns"]:
                        turn_content = f"{turn['speaker']}: {turn['text']}"
                        vector = await get_embedding(turn_content)
                        await cur.execute(
                            "INSERT INTO child_chunks (parent_id, content, embedding) VALUES (%s, %s, %s)",
                            (parent_id, turn_content, vector),
                        )

        return {
            "status": "success",
            "parents_processed": len(parent_blocks),
            "chunks_processed": sum(len(b["turns"]) for b in parent_blocks),
            "message": f"Successfully processed {len(parent_blocks)} parent documents and embedded child chunks.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExtractionQuery(BaseModel):
    query: str
    limit: int = 3


@app.post("/extract_actions")
async def extract_action_items(payload: ExtractionQuery):
    try:
        # Step 1: Embed the search query using Gemini embeddings
        query_vector = await get_embedding(payload.query)

        # Step 2: Retrieve the closest child chunks from child_chunks table (high precision)
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT parent_id 
                    FROM child_chunks 
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """,
                    (query_vector, max(payload.limit, 5)),  # Get top 5 child chunks to ensure rich context
                )
                child_results = await cur.fetchall()

                if not child_results:
                    return {
                        "status": "empty",
                        "message": "No relevant context found in memory.",
                    }

                # Step 3: Resolve unique parent IDs using stable insertion-ordered deduplication
                parent_ids = list(dict.fromkeys(parent_id for (parent_id,) in child_results))

                # Step 4: Fetch complete parent documents (substitution/Small-to-Big expansion)
                await cur.execute(
                    """
                    SELECT id, topic, content 
                    FROM parent_documents 
                    WHERE id = ANY(%s)
                """,
                    (parent_ids,),
                )
                parent_results = await cur.fetchall()

                # Step 5: Preserve the original relevance order of parent documents
                parent_map = {row[0]: (row[1], row[2]) for row in parent_results}
                ordered_results = [parent_map[pid] for pid in parent_ids if pid in parent_map]

        # Format the retrieved parent contexts into a single string for Gemini
        context_string = "\n\n".join([f"[{row[0]}]\n{row[1]}" for row in ordered_results])

        # Step 3: Orchestrate Gemma 4 to extract structured action items
        system_prompt = """
        You are OmniScribe, an autonomous Process Manager for a senior engineering team.
        Read the provided meeting transcript context and extract ONLY the concrete action items that are directly relevant to the user's query. Do not extract tasks that are unrelated to the query. If no tasks in the context are relevant to the query, return an empty "action_items" list.
        
        You must respond ONLY with a valid JSON object containing an "action_items" list of tasks. Do not include markdown formatting like ```json.
        Each task object in the list must have three keys:
        - "assignee": The person tasked with the item.
        - "task": A clear, concise description of the architecture or engineering task.
        - "status": Always set this to "PENDING".
        """

        response = await reasoning_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_string}\n\nExtract the action items based on my query: {payload.query}",
                },
            ],
            temperature=0.1,  # Keep it highly deterministic
            response_format={"type": "json_object"},  # Force structured output
        )

        # Parse the raw JSON string returned by Gemma
        raw_output = response.choices[0].message.content.strip()
        extracted_data = json.loads(raw_output)
        action_items = extracted_data.get("action_items", [])

        # --- Save to Database ---
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                for item in action_items:
                    await cur.execute(
                        "INSERT INTO action_items (assignee, task, status) VALUES (%s, %s, %s)",
                        (item.get("assignee"), item.get("task"), "PENDING")
                    )
        # -----------------------------

        return {
            "status": "success",
            "context_retrieved": len(ordered_results),
            "action_items": action_items,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_slot_data():
    """Background generator that constantly polls llama.cpp and yields the data."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Ping the local Gemma 4 server
                response = await client.get("http://localhost:11434/slots")
                data = response.json()

                # Format the payload for Server-Sent Events (must start with 'data: ')
                yield f"data: {json.dumps(data)}\n\n"

            except Exception:
                yield f"data: {json.dumps({'error': 'Engine offline'})}\n\n"

            # Wait 500ms before pushing the next update
            await asyncio.sleep(0.5)


@app.get("/stream_slots")
async def stream_slots():
    """The endpoint the frontend connects to via EventSource."""
    return StreamingResponse(fetch_slot_data(), media_type="text/event-stream")


@app.post("/execute_tasks")
async def execute_pending_tasks():
    try:
        # 1. Find all pending tasks
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id, assignee, task FROM action_items WHERE status = 'PENDING'")
                pending_tasks = await cur.fetchall()
                
                if not pending_tasks:
                    return {"status": "idle", "message": "No pending tasks to execute."}

                executed_log = []
                
                # 2. Simulate the AI "Execution" (e.g., calling the Jira API)
                for task_id, assignee, task in pending_tasks:
                    # Simulate non-blocking network latency for API calls
                    await asyncio.sleep(1) 
                    
                    action_log = f"Successfully created Jira ticket for {assignee}: '{task}'"
                    executed_log.append(action_log)
                    
                    # 3. Mark as completed in the database
                    await cur.execute("UPDATE action_items SET status = 'COMPLETED' WHERE id = %s", (task_id,))
                
        return {
            "status": "success",
            "tasks_completed": len(pending_tasks),
            "execution_log": executed_log
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
