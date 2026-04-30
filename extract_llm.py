import os
import json
from groq import Groq

def extract_llm(sentence):  
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found.")
        return []
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are a Knowledge Graph extraction system. Extract strict Subject-Relation-Object triples from the text.
    RULES:
    1. Relations MUST be core actions or verbs (e.g., "acquired", "is", "serves as CEO of").
    2. DO NOT extract prepositions like "in", "on", "at", or "by" as relations.
    3. Return ONLY a valid JSON array of objects. Do not include markdown formatting, explanations, or text outside the JSON.
    Format: [{{"subject": "...", "relation": "...", "obj": "..."}}]
    
    Text: "{sentence}"
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", 
            temperature=0.0
        )
        
        content = response.choices[0].message.content.strip()
        
        
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
            
        triples = json.loads(content)
        
        
        valid_triples = []
        for t in triples:
            if "subject" in t and "relation" in t and "obj" in t:
                valid_triples.append({
                    "subject": t["subject"],
                    "relation": t["relation"],
                    "obj": t["obj"],
                    "sentence": sentence,
                    "source": "llm"
                })
        return valid_triples
    except Exception as e:
        print(f"LLM Extraction failed for sentence: '{sentence[:30]}...' Error: {e}")
        return []