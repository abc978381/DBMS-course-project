from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_rebel_tokenizer = None
_rebel_model = None

def load_rebel():
    global _rebel_tokenizer, _rebel_model
    if _rebel_model is None:
        print("Loading REBEL model (this might take a moment)...")
        model_name = 'Babelscape/rebel-large'
        _rebel_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _rebel_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _rebel_tokenizer, _rebel_model

def parse_rebel_output(text):
    triplets = []
    relation, subject, object_ = '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'relation': relation.strip(), 'obj': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'relation': relation.strip(), 'obj': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't': subject += ' ' + token
            elif current == 's': object_ += ' ' + token
            elif current == 'o': relation += ' ' + token
    
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'subject': subject.strip(), 'relation': relation.strip(), 'obj': object_.strip()})
    return triplets

def extract_rebel(sentence):
    tokenizer, model = load_rebel()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
    
    extracted_ids = model.generate(
        inputs["input_ids"], 
        max_length=256, 
        num_beams=3, 
        length_penalty=0, 
        early_stopping=True
    )
    
    decoded_text = tokenizer.batch_decode(extracted_ids, skip_special_tokens=False)[0]
    raw_triples = parse_rebel_output(decoded_text)
    formatted_triples = []
    for t in raw_triples:
        t["sentence"] = sentence
        t["source"] = "rebel"
        formatted_triples.append(t)
    return formatted_triples