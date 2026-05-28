import httpx

# A rich architecture transcript containing specific tasks, assignees, and technology stacks
transcript = """
Alex: We need to figure out the outbox pattern. Right now we are dropping events.
Sarah: I will write the outbox publisher in Rust.
Alex: What about the inbox consumer?
Sarah: Dave volunteered to build the inbox consumer using Go.
Dave: Yes, I can start on the Go inbox consumer next week.
Sarah: We also need to configure PostgreSQL indexing for high-volume message queues.
Alex: I can take the indexing task. I will write the migration script in SQL.
Sarah: Perfect. Let's make sure we have monitoring.
Alex: We can set up Prometheus. Who can do that?
Sarah: I can handle Prometheus setup.
"""

def seed():
    try:
        response = httpx.post("http://localhost:8001/ingest", json={
            "topic": "Architecture Meeting",
            "transcript": transcript
        }, timeout=20.0)
        
        if response.status_code == 200:
            print("Database successfully seeded with the Golden Dataset transcript!")
            print("Response:", response.json())
        else:
            print(f"Failed to seed database. HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print("Error during database seeding:", str(e))

if __name__ == "__main__":
    seed()
