import spacy
import re
import json

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def segment_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

def normalize_text(text, is_relation=False):
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)   
    text = text.strip().lower()
    
    if is_relation:
        doc = nlp(text)
        lemmas = []
        for token in doc:
            if token.pos_ not in ("AUX", "PART", "ADP"):
                lemmas.append(token.lemma_)
        if not lemmas:
            return text
        return " ".join(lemmas)
    return text

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Safely loads and parses a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)