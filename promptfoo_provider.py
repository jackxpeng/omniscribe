import httpx
import json

def call_api(prompt, options, context):
    # Grab the user's query from the promptfoo test case
    vars = context.get('vars', {})
    query = vars.get('query')
    
    try:
        # Hit your running FastAPI server
        response = httpx.post("http://localhost:8001/extract_actions", json={
            "query": query,
            "limit": 3
        }, timeout=30.0) # Give Gemini time to think
        
        response.raise_for_status()
        data = response.json()
        
        # promptfoo expects a string output to evaluate
        output = json.dumps(data.get("action_items", []))
        
        return {
            "output": output,
        }
    except Exception as e:
        return {
            "error": str(e)
        }
