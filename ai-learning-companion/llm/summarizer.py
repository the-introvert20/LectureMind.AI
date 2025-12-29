import os
import re
from typing import List, Dict, Tuple
import json


def _improved_fallback_notes(transcript: str) -> str:
    """Enhanced fallback with better structure and content analysis."""
    if not transcript or not transcript.strip():
        return "# Lecture Notes\n\nNo transcript content available."

    # Basic text cleaning and sentence splitting
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript) if s.strip()]
    
    # Identify potential headers (sentences that might be headings)
    potential_headers = [s for s in sentences 
                        if len(s.split()) <= 12 
                        and not s.endswith(('.', '!', '?')) 
                        and not s.islower()]
    
    # Create basic structure
    md = ["# Lecture Notes", ""]
    
    # Add key points section
    md.extend(["## Key Points", ""])
    for s in sentences[:min(8, len(sentences))]:
        if s not in potential_headers:  # Don't duplicate headers as bullet points
            md.append(f"- {s}")
    
    # Add detailed notes section
    md.extend(["", "## Detailed Notes", ""])
    
    # Group related sentences into paragraphs
    current_paragraph = []
    for s in sentences[8:]:
        if s in potential_headers:
            if current_paragraph:
                md.append(" ".join(current_paragraph) + "\n")
                current_paragraph = []
            md.append(f"### {s}\n")
        else:
            current_paragraph.append(s)
    
    if current_paragraph:
        md.append(" ".join(current_paragraph))
    
    return "\n".join(md)

def _multi_pass_processing(transcript: str, concepts: List[str] = None) -> str:
    """Process transcript through multiple stages for better note quality."""
    # First pass: Extract key concepts and relationships
    concepts = concepts or []
    
    # Second pass: Organize content hierarchically
    structure_prompt = f"""Analyze this lecture transcript and create a hierarchical outline.
    Focus on identifying main topics, subtopics, and key points.
    Pay attention to: {', '.join(concepts) if concepts else 'all major concepts'}.
    
    Return a JSON structure with:
    - main_topics: List of main topics with their subtopics and key points
    - key_definitions: Important terms and their definitions
    - examples: Key examples or case studies
    - relationships: How concepts relate to each other
    
    Transcript:
    {transcript}"""
    
    # Third pass: Generate well-formatted notes
    notes_prompt = """Create comprehensive, well-structured study notes using this outline:
    {outline}
    
    Follow these guidelines:
    1. Use clear, hierarchical headings (## for main topics, ### for subtopics)
    2. Include bullet points for key details
    3. Add numbered lists for processes or sequences
    4. Use **bold** for important terms and definitions
    5. Include examples where helpful
    6. Add section summaries for better retention
    
    Format the output in Markdown."""
    
    return structure_prompt + "\n\n" + notes_prompt

def _post_process_notes(notes: str) -> str:
    """Clean and format the generated notes."""
    if not notes:
        return ""
    
    # Remove duplicate sections
    seen = set()
    lines = []
    for line in notes.split('\n'):
        clean_line = line.strip()
        if clean_line.startswith('#') and clean_line in seen:
            continue
        seen.add(clean_line)
        lines.append(line)
    
    # Ensure consistent heading levels
    notes = '\n'.join(lines)
    
    # Remove redundant information
    notes = re.sub(r'\n{3,}', '\n\n', notes)  # Remove excessive newlines
    
    return notes.strip()

def generate_notes(transcript: str, concepts: List[str] = None) -> str:
    """Generate high-quality, structured study notes from lecture transcript.
    
    Args:
        transcript: The raw lecture transcript text
        concepts: Optional list of key concepts to focus on
        
    Returns:
        Formatted Markdown string with structured notes
    """
    # Check for local endpoint first, then cloud
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1") # Default to local Ollama
    model = os.environ.get("OPENAI_MODEL", "gemma3:1b") # Default to Gemma 3 1B
    
    is_local = "localhost" in api_base or "127.0.0.1" in api_base

    if not transcript or not transcript.strip():
        return "# Lecture Notes\n\nNo transcript content provided."

    if api_key or is_local:
        try:
            import requests
            
            # Generate the enhanced prompt
            prompt = _multi_pass_processing(transcript, concepts)
            
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # First generate the structure
            structure_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are an expert at analyzing educational content and creating structured outlines."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
            }
            
            resp = requests.post(
                f"{api_base}/chat/completions", 
                headers=headers, 
                json=structure_payload, 
                timeout=120
            )
            resp.raise_for_status()
            structure = resp.json()["choices"][0]["message"]["content"]
            
            # Then generate the final notes
            notes_prompt = f"""Using this lecture analysis, create comprehensive study notes:
            
            {structure}
            
            Follow these formatting guidelines:
            - Use clear, hierarchical headings (## Main Topic, ### Subtopic)
            - Include bullet points for key details
            - Use numbered lists for processes or sequences
            - Add **bold** for important terms and definitions
            - Include examples in blockquotes
            - End with a summary of key takeaways
            """
            
            notes_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a professional note-taker creating clear, concise, and well-organized study notes."
                    },
                    {"role": "user", "content": notes_prompt}
                ],
                "temperature": 0.3,
            }
            
            resp = requests.post(
                f"{api_base}/chat/completions", 
                headers=headers, 
                json=notes_payload, 
                timeout=120
            )
            resp.raise_for_status()
            
            raw_notes = resp.json()["choices"][0]["message"]["content"]
            
            # Post-process the notes
            return _post_process_notes(raw_notes)
            
        except Exception as e:
            print(f"AI generation failed: {e}. Falling back to improved heuristics.")
            return _improved_fallback_notes(transcript)
    else:
        return _improved_fallback_notes(transcript)