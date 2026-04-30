import re
from utils import normalize_text, nlp
from sentence_transformers import SentenceTransformer, util

_sem_model = None

def get_sem_model():
    global _sem_model
    if _sem_model is None:
        print("Loading Semantic Model for Fusion...")
        _sem_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sem_model

def calculate_lexical_overlap(triple, sentence):
    """Calculates extraction fidelity by measuring the overlap of core informational words."""
    sentence_doc = nlp(sentence.lower())
    sentence_core_words = {
        token.lemma_ for token in sentence_doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    }
    triple_text = f"{triple['subject']} {triple['relation']} {triple['obj']}".lower()
    triple_doc = nlp(triple_text)
    triple_core_words = {
        token.lemma_ for token in triple_doc 
        if not token.is_stop and not token.is_punct and token.text.strip()
    }
    if not triple_core_words:
        return 0.0
    overlap = triple_core_words.intersection(sentence_core_words)
    return len(overlap) / len(triple_core_words)

def calculate_semantic_similarity(triple, sentence):
    model = get_sem_model()
    triple_text = f"{triple['subject']} {triple['relation']} {triple['obj']}"
    embs = model.encode([triple_text, sentence], convert_to_tensor=True)
    sim = float(util.cos_sim(embs[0], embs[1])[0][0])
    return max(0.0, min(1.0, sim))

def calculate_harmonic_confidence(triple, sentence):
    """F1-style harmonic mean of Lexical Grounding and Semantic Meaning."""
    lex = calculate_lexical_overlap(triple, sentence)
    sem = calculate_semantic_similarity(triple, sentence)
    if lex + sem == 0: 
        return 0.0
    return round(2 * lex * sem / (lex + sem), 3)

def clean_entity_text(text):
    clean_text = text.split(',')[0].strip()
    clean_text = re.sub(r'^(?i)(a|an|the)\s+', '', clean_text).strip()
    return clean_text

def fuse_triples(spacy_triples, llm_triples):
    fused_dict = {}
    def process_triples(triples, source_name):
        for t in triples:
            t['subject'] = clean_entity_text(t['subject'])
            t['obj'] = clean_entity_text(t['obj'])
            lemmatized_relation = normalize_text(t['relation'], is_relation=True)
            key = (normalize_text(t['subject']), lemmatized_relation, normalize_text(t['obj']))
            if not all(key): continue 
            confidence = calculate_harmonic_confidence(t, t['sentence'])
            
            if key in fused_dict:
                current_conf = fused_dict[key]['confidence']
                fused_dict[key]['confidence'] = max(current_conf, confidence)
                fused_dict[key]['source'] = 'hybrid'
                if source_name == 'llm':
                    fused_dict[key]['relation'] = t['relation'] 
            else:
                t['confidence'] = confidence
                t['source'] = source_name
                fused_dict[key] = t

    process_triples(spacy_triples, 'spacy')
    process_triples(llm_triples, 'llm')

    return list(fused_dict.values())