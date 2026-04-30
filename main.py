import os
import sys
from coref_resolver import resolve_coreferences
from dotenv import load_dotenv
from utils import segment_sentences, save_json, load_json
from extract_spacy import extract_spacy
from extract_llm import extract_llm
from extract_rebel import extract_rebel
from fusion import fuse_triples
from evaluate import evaluate_extraction
from neo4j_loader import Neo4jLoader
import pandas as pd

load_dotenv()

def read_input_file(filepath="input.txt"):
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: '{filepath}' not found.")
        print(f"Please create an '{filepath}' file in the same directory as this script.")
        sys.exit(1)
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().strip()
        
    if not text:
        print(f"CRITICAL ERROR: '{filepath}' is empty. Please add some text.")
        sys.exit(1)
        
    return text

def main():
    print(" Starting Knowledge Graph Pipeline ")
    
    raw_input_text = read_input_file("input.txt")
    ground_truth_data = load_json("ground_truth.json")

    print("Resolving Coreferences...")
    resolved_text = resolve_coreferences(raw_input_text)

    sentences = segment_sentences(resolved_text)
    print(f"Loaded input.txt and segmented into {len(sentences)} sentences.")
    
    print("\nRunning Extractions...")
    spacy_results = []
    llm_results = []
    rebel_results = []
    
    for sent in sentences:
        spacy_results.extend(extract_spacy(sent))
        llm_results.extend(extract_llm(sent))
        rebel_results.extend(extract_rebel(sent))

    print("Fusing Hybrid Model (spaCy + LLM)...")
    hybrid_results = fuse_triples(spacy_results, llm_results)
    
    save_json(hybrid_results, "final_triples.json")
    print("Saved 'final_triples.json'")

    print("\nLet's Evaluate Models...")
    evals = {
        "spaCy": evaluate_extraction(spacy_results, ground_truth_data),
        "REBEL": evaluate_extraction(rebel_results, ground_truth_data),
        "Hybrid (spaCy+LLM)": evaluate_extraction(hybrid_results, ground_truth_data)
    }
    
    df = pd.DataFrame(evals).T
    print("\n--- EXPERIMENT RESULTS ---")
    print(df.to_string())
    print("--------------------------\n")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    if uri and user and password:
        print("Loading Hybrid Results into Neo4j...")
        try:
            loader = Neo4jLoader(uri, user, password)
            loader.setup_schema()
            loader.load_triples(hybrid_results)
            loader.close()
            print("Graph pipeline complete! Check your Neo4j browser.")
        except Exception as e:
            print(f"Neo4j Connection Error: {e}\nEnsure your database is running and credentials in .env are correct.")
    else:
        print("Skipping Neo4j Upload. Missing .env variables.")

if __name__ == "__main__":
    main()