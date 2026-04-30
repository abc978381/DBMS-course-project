import spacy
from utils import nlp

def extract_spacy(sentence):
    
    doc = nlp(sentence)
    triples = []
    
    for token in doc:
        
        if token.pos_ == "VERB":
            subject, obj = None, None           
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                    subject = " ".join([w.text for w in child.subtree])
                if child.dep_ in ("dobj", "pobj", "attr", "acomp"):
                    obj = " ".join([w.text for w in child.subtree])
            
            if subject and obj:
                triples.append({
                    "subject": subject,
                    "relation": token.text,
                    "obj": obj,
                    "sentence": sentence,
                    "source": "spacy"
                })
    return triples