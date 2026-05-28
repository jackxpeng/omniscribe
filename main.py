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


from utils import parse_transcript_to_turns, chunk_by_speaker

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
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS meeting_memory (
                        id SERIAL PRIMARY KEY,
                        topic VARCHAR(255),
                        content TEXT,
                        embedding VECTOR(768)
                    );
                """)
            print("Database initialized successfully.")
    except Exception as e:
        print(f"Database initialization failed: {e}")
        
    yield
    
    # Clean shutdown of connection pool
    await app.state.db_pool.close()

app = FastAPI(title="OmniScribe Process Manager", lifespan=lifespan)

# 1. Initialize the Local Embedding Client (Pointing to Nomic on Port 11435)
embedding_client = AsyncOpenAI(
    base_url="http://localhost:11435/v1",
    api_key="sk-no-key-required",  # Local servers don't need real keys
)

reasoning_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required"
)


class TranscriptPayload(BaseModel):
    topic: str
    transcript: str


async def get_embedding(text: str) -> list[float]:
    """Calls our local Nomic microservice to convert text to a vector."""
    response = await embedding_client.embeddings.create(
        input=text,
        model="nomic-embed-text",  # Name doesn't matter for llama.cpp, but required by the API schema
    )
    return response.data[0].embedding


@app.post("/ingest")
async def ingest_transcript(payload: TranscriptPayload):
    try:
        # Step 1: Semantic Chunking
        turns = parse_transcript_to_turns(payload.transcript)
        chunks = chunk_by_speaker(turns, max_chars=1500)

        if not chunks:
            raise HTTPException(
                status_code=400, detail="Transcript could not be parsed into chunks."
            )

        # Step 2 & 3: Generate Vectors and Insert to DB
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk in chunks:
                    # Ask Port 11435 for the vector
                    vector = await get_embedding(chunk)

                    # Save the text and the vector together
                    await cur.execute(
                        "INSERT INTO meeting_memory (topic, content, embedding) VALUES (%s, %s, %s)",
                        (payload.topic, chunk, vector),
                    )

        return {
            "status": "success",
            "chunks_processed": len(chunks),
            "message": f"Successfully embedded and saved {len(chunks)} chunks to memory.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExtractionQuery(BaseModel):
    query: str
    limit: int = 3


@app.post("/extract_actions")
async def extract_action_items(payload: ExtractionQuery):
    try:
        # Step 1: Embed the search query using Nomic
        query_vector = await get_embedding(payload.query)

        # Step 2: Retrieve the closest context from PostgreSQL
        # The <=> operator calculates Cosine Distance in pgvector
        async with app.state.db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT topic, content 
                    FROM meeting_memory 
                    ORDER BY embedding <=> %s::vector 
                    LIMIT %s
                """,
                    (query_vector, payload.limit),
                )
                results = await cur.fetchall()

        if not results:
            return {
                "status": "empty",
                "message": "No relevant context found in memory.",
            }

        # Format the retrieved context into a single string for Gemma
        context_string = "\n\n".join([f"[{row[0]}]\n{row[1]}" for row in results])

        # Step 3: Orchestrate Gemma 4 to extract structured action items
        system_prompt = """
        You are OmniScribe, an autonomous Process Manager for a senior engineering team.
        Read the provided meeting transcript context and extract any concrete action items.
        
        You must respond ONLY with a valid JSON object containing an "action_items" list of tasks. Do not include markdown formatting like ```json.
        Each task object in the list must have three keys:
        - "assignee": The person tasked with the item.
        - "task": A clear, concise description of the architecture or engineering task.
        - "status": Always set this to "PENDING".
        """

        response = await reasoning_client.chat.completions.create(
            model="gemma-4",  # Name ignored by llama.cpp
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

        return {
            "status": "success",
            "context_retrieved": len(results),
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

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
