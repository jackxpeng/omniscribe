import re
from typing import List, Dict

def parse_transcript_to_turns(transcript_text: str) -> List[Dict[str, str]]:
    """
    Parses a raw transcript string into a list of structured speaker turns.
    Assumes standard format like "Alex: Let's use PostgreSQL."
    """
    # Regex looks for a name (letters/spaces) followed by a colon at the start of a line
    pattern = re.compile(r"^([A-Za-z\s]+):\s*(.*)", re.MULTILINE)
    
    turns = []
    last_end = 0
    
    # Find all speaker declarations
    matches = list(pattern.finditer(transcript_text))
    
    for i, match in enumerate(matches):
        speaker = match.group(1).strip()
        
        # The text for this turn goes from the colon to the start of the NEXT speaker
        start_idx = match.end(1) + 1 # Skip the colon
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(transcript_text)
        
        text = transcript_text[start_idx:end_idx].strip()
        
        if text:
            turns.append({"speaker": speaker, "text": text})
            
    return turns

def chunk_by_speaker(turns: List[Dict[str, str]], max_chars: int = 1000) -> List[str]:
    """
    Aggregates speaker turns into larger chunks for embedding.
    Ensures chunks do not exceed max_chars unless a single turn is massive.
    """
    chunks = []
    current_chunk_text = ""
    
    for turn in turns:
        # Format the turn back into readable text for the LLM
        turn_text = f"{turn['speaker']}: {turn['text']}\n"
        
        # If adding this turn exceeds our limit, save the current chunk and start a new one
        if len(current_chunk_text) + len(turn_text) > max_chars and current_chunk_text:
            chunks.append(current_chunk_text.strip())
            current_chunk_text = turn_text
        else:
            current_chunk_text += turn_text
            
    # Catch the final chunk
    if current_chunk_text:
        chunks.append(current_chunk_text.strip())
        
    return chunks

# --- Quick Local Test ---
if __name__ == "__main__":
    sample_text = """
    Alex: We need to figure out the outbox pattern. 
    Right now we are dropping events.
    Sarah: I can build a polling publisher. 
    Dave: Wait, isn't polling too slow?
    Sarah: Not if we tune the indexing on the Postgres table.
    """
    
    turns = parse_transcript_to_turns(sample_text)
    # Using a tiny max_chars to force it to split for the demonstration
    chunks = chunk_by_speaker(turns, max_chars=100) 
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")