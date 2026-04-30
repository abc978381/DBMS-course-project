# Populating a Knowledge Graph from Natural Language Text

An end-to-end NLP pipeline that extracts Open Information Extraction (OpenIE) triples from unstructured text, mathematically fuses cross-architecture results, evaluates semantic accuracy, and populates a Neo4j Knowledge Graph.

## 🚀 Features

* **Advanced Preprocessing:** Uses `fastcoref` for neural coreference resolution, including custom logic for possessive pronoun reconstruction.
* **Hybrid Extraction Engine:** 
  * **spaCy:** Rule-based dependency parsing for high-precision linguistic extraction.
  * **Groq LLM (Llama 3.1 8B):** Generative zero-shot extraction for high-recall semantic extraction, handling conjunctions and complex predicates.
  * **REBEL:** Closed-ontology Seq2Seq baseline.
* **Intelligent Fusion:** Mathematically merges overlapping triples using a Harmonic Mean of **Lexical Fidelity** (preventing hallucinations) and **Semantic Cosine Similarity** (`all-MiniLM-L6-v2`), completely eliminating arbitrary confidence heuristics.
* **Graph Database Upsertion:** Idempotent Cypher generation with schema uniqueness constraints loaded directly into a local Neo4j instance.

## 📂 Project Structure
```text
├── coref_resolver.py     # Neural pronoun resolution
├── evaluate.py           # Harmonic F1 Semantic Evaluation
├── extract_llm.py        # Generative OpenIE via Groq API
├── extract_rebel.py      # Closed-ontology Seq2Seq baseline
├── extract_spacy.py      # Rule-based dependency parsing
├── fusion.py             # Math-based deduplication and confidence scoring
├── main.py               # Pipeline orchestrator
├── neo4j_loader.py       # Graph DB connection and Cypher querying
├── utils.py              # NLP normalizers, POS tagging, and file I/O
├── input.txt             # Target unstructured text (User provided)
├── ground_truth.json     # Evaluation benchmark data (User provided)
├── requirements.txt      # Pinned dependency map
└── .env                  # API keys and DB credentials

🛠️ Installation & Setup
1. Clone the repository and navigate to the directory

Bash
git clone [https://github.com/yourusername/knowledge-graph-pipeline.git](https://github.com/yourusername/knowledge-graph-pipeline.git)
cd knowledge-graph-pipeline
2. Create a Virtual Environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies

Bash
pip install -r requirements.txt
(Note: spaCy's English model en_core_web_sm will automatically download on the first run via utils.py if not detected).

4. Environment Variables
Create a .env file in the root directory with the following keys:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_database_password
5. Start Neo4j
Ensure your local Neo4j desktop or Docker container is running and matches the credentials in your .env file.

🧠 Usage
Paste your unstructured text into input.txt.

(Optional) Update ground_truth.json with your expected triples to benchmark the F1 score.

Run the orchestrator:

Bash
python main.py
The console will output the segmentation details, run the extraction experiments, print a Pandas DataFrame comparing the precision/recall/F1 metrics, and finally upload the hybrid corroborated graph to Neo4j. You can view the graph by opening your Neo4j Browser.

📊 Evaluation Strategy
This pipeline moves away from brittle string-matching and arbitrary confidence thresholds. Fused triples are evaluated using a Harmonic Mean:

Lexical Overlap: Measures the strict intersection of informational root lemmas between the extracted tuple and the source text (acting as a "Hallucination Guillotine").

Semantic Similarity: Computes the cosine similarity of dense vector embeddings to validate contextual meaning.

Triples must possess both exactness and meaning to be highly scored and uploaded to the graph.