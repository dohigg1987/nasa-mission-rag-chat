# NASA Mission Intelligence: RAG Chat System

**Author:** Dennis O'Higgins

A Retrieval-Augmented Generation (RAG) chat system that answers questions about the Apollo 11, Apollo 13 and Challenger (STS-51-L) missions using NASA mission transcripts and technical documents. It chunks and embeds the documents into ChromaDB, retrieves the most relevant excerpts for each question, generates a cited answer with an OpenAI chat model, and scores every answer in real time with RAGAS.

## Project structure

| File | Purpose |
|---|---|
| `embedding_pipeline.py` | Chunks the NASA text files, creates OpenAI embeddings and stores them with metadata in ChromaDB (command-line tool). |
| `rag_client.py` | Discovers ChromaDB collections, runs semantic search (optionally filtered by mission) and formats the retrieved chunks as cited context. |
| `llm_client.py` | NASA-expert system prompt, conversation history management and the OpenAI chat call. |
| `ragas_evaluator.py` | Scores a (question, context, answer) triple with RAGAS: response relevancy, faithfulness, context precision, plus BLEU and ROUGE when a reference answer is given. |
| `chat.py` | Streamlit chat application that brings the components together and shows the scores for each answer. |
| `batch_evaluation.py` | Runs every question in `evaluation_dataset.txt` end to end and prints per-question scores and per-metric aggregates. |
| `evaluation_dataset.txt` | Seven test questions across overview, emergency, disaster analysis, crew, technical and timeline categories, with expected answers. |
| `fix_ragas_imports.py` | Applies the RAGAS 0.4.3 VertexAI import fix described in the course instructions. |

The NASA source documents are in `data_text/` (apollo11, apollo13, challenger) in the course starter repository. The ChromaDB folder is created by the pipeline and is not stored in this repository.

## Setup

```bash
pip install -r requirements.txt
python fix_ragas_imports.py          # one-off fix so that `import ragas` works with RAGAS 0.4.3
export OPENAI_API_KEY="your-key"     # Udacity/Vocareum keys (voc-...) are routed to the Vocareum endpoint automatically
```

## 1. Build the vector database

```bash
python embedding_pipeline.py --data-path ./data_text \
    --chroma-dir ./chroma_db_openai --collection-name nasa_space_missions_text \
    --chunk-size 500 --chunk-overlap 100 --batch-size 100 --update-mode skip
```

* `--chunk-size` / `--chunk-overlap` (characters) are set at runtime. Chunks never exceed `chunk_size`, prefer to end at a sentence or line break, and consecutive chunks always overlap by exactly `chunk_overlap` characters.
* Each chunk is embedded with `text-embedding-3-small` and stored with metadata including `source`, `file_path`, `mission` (apollo_11, apollo_13, challenger), `document_category`, `chunk_index` and character offsets.
* `--update-mode skip|update|replace` controls documents that are already stored (stable IDs of the form `mission_source_chunk_0001`).
* Inspect the collection without re-embedding:

```bash
python embedding_pipeline.py --stats-only --chroma-dir ./chroma_db_openai --collection-name nasa_space_missions_text
```

## 2. Run the chat application

```bash
streamlit run chat.py
```

Choose the collection, model, number of chunks to retrieve (k) and an optional mission focus in the sidebar. Each answer cites its sources, the retrieved excerpts can be expanded under the answer, and the RAGAS scores appear in the sidebar.

## 3. Batch evaluation

```bash
python batch_evaluation.py --dataset evaluation_dataset.txt --k 5
```

Prints the answer and scores for each question and the mean, minimum and maximum of every metric, and saves `evaluation_results.json` and `evaluation_results.csv`.
