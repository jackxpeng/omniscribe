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

def partition_turns_to_parents(turns: List[Dict[str, str]], max_chars: int = 1500) -> List[Dict[str, any]]:
    """
    Groups speaker turns chronologically into parent blocks (up to max_chars),
    retaining the list of individual child turns that belong to each parent.
    Returns a list of dictionaries: {"content": str, "turns": list}
    """
    parent_blocks = []
    current_parent_turns = []
    current_content = ""

    for turn in turns:
        turn_text = f"{turn['speaker']}: {turn['text']}\n"
        if len(current_content) + len(turn_text) > max_chars and current_content:
            parent_blocks.append({
                "content": current_content.strip(),
                "turns": current_parent_turns
            })
            current_content = turn_text
            current_parent_turns = [turn]
        else:
            current_content += turn_text
            current_parent_turns.append(turn)

    if current_content:
        parent_blocks.append({
            "content": current_content.strip(),
            "turns": current_parent_turns
        })
        
    return parent_blocks


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
    blocks = partition_turns_to_parents(turns, max_chars=100) 
    
    for i, block in enumerate(blocks):
        print(f"--- Parent {i+1} ---\n{block['content']}\n")
        print("Children:")
        for turn in block["turns"]:
            print(f"  - {turn['speaker']}: {turn['text']}")
        print()