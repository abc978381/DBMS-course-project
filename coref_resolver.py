from fastcoref import FCoref

_coref_model = None

POSSESSIVE_PRONOUNS = {"his", "her", "hers", "its", "their", "theirs", "my", "our", "your"}

def get_coref_model():
    global _coref_model
    if _coref_model is None:
        print("Loading FastCoref model...")
        _coref_model = FCoref(device='cpu') 
    return _coref_model

def resolve_coreferences(text):
    model = get_coref_model()
    preds = model.predict(texts=[text])
    clusters = preds[0].get_clusters(as_strings=False) 
    if not clusters:
        return text
    replacements = []
    
    for cluster in clusters:
        cluster = sorted(cluster, key=lambda x: x[0])
        
        root_span = cluster[0]
        root_entity = text[root_span[0]:root_span[1]]
        
        for mention_span in cluster[1:]:
            mention_text = text[mention_span[0]:mention_span[1]]
            replacement_text = root_entity
            
            if mention_text.lower() in POSSESSIVE_PRONOUNS:
                if not replacement_text.endswith("'s") and not replacement_text.endswith("'"):
                    replacement_text += "'s"
                    
            replacements.append({
                "start": mention_span[0], 
                "end": mention_span[1], 
                "text": replacement_text
            })

    replacements.sort(key=lambda x: x["start"], reverse=True)
    resolved_text = text
    for rep in replacements:
        resolved_text = resolved_text[:rep["start"]] + rep["text"] + resolved_text[rep["end"]:]
    return resolved_text