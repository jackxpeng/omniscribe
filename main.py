from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector


from utils import parse_transcript_to_turns, chunk_by_speaker

app = FastAPI(title="OmniScribe Process Manager")

# 1. Initialize the Local Embedding Client (Pointing to Nomic on Port 11435)
embedding_client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="sk-no-key-required" # Local servers don't need real keys
)


# Initialize the schema
try:
    # Connect to the local pgvector container
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # 3072 dimensions is the standard output for large embedding models
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meeting_memory (
                id SERIAL PRIMARY KEY,
                topic VARCHAR(255),
                content TEXT,
                embedding VECTOR(768)
            );
        """)
        register_vector(conn)
except Exception as e:
    print(f"Database connection failed: {e}")


class TranscriptPayload(BaseModel):
    topic: str
    transcript: str


def get_embedding(text: str) -> list[float]:
    """Calls our local Nomic microservice to convert text to a vector."""
    response = embedding_client.embeddings.create(
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
        with conn.cursor() as cur:
            for chunk in chunks:
                # Ask Port 11435 for the vector
                vector = get_embedding(chunk)

                # Save the text and the vector together
                cur.execute(
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
