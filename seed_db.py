import httpx

# A realistic dataset containing one target meeting transcript and three distinct distractor transcripts
# to verify that parent-child retrieval fetches the correct parent document among many.
transcripts = [
    {
        "topic": "Architecture Meeting",
        "transcript": """
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
    },
    {
        "topic": "Q3 Marketing Campaign",
        "transcript": """
John: We need to launch the ad campaign by next Friday.
Alice: I will prepare the ad copy and graphics.
John: Great, what about social media outreach?
Bob: I can handle the Twitter and LinkedIn posts.
John: Excellent, let's keep track of our budgets in Excel.
"""
    },
    {
        "topic": "Quarterly Security Audit",
        "transcript": """
Rachel: We need to perform the penetration testing on our staging environment.
Tom: I will run the security scanning tools next Tuesday.
Rachel: Are we checking our AWS S3 bucket permissions?
Tom: Yes, I can review the IAM policies.
Rachel: Perfect, let's document all findings in Confluence.
"""
    },
    {
        "topic": "Mobile App Exploration",
        "transcript": """
Emma: We should think about a mobile app.
Liam: I can research React Native next year.
Emma: Good, let's wait until we finish the web dashboard.
"""
    }
]

def seed():
    try:
        for t in transcripts:
            response = httpx.post("http://localhost:8001/ingest", json={
                "topic": t["topic"],
                "transcript": t["transcript"]
            }, timeout=20.0)
            
            if response.status_code == 200:
                print(f"Successfully seeded transcript for '{t['topic']}'!")
                print("Response:", response.json())
            else:
                print(f"Failed to seed '{t['topic']}'. HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print("Error during database seeding:", str(e))

if __name__ == "__main__":
    seed()
