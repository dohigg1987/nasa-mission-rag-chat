# NASA Mission Intelligence RAG System: Project Report

**Author:** Dennis O'Higgins

## 1. Overview

This project builds a Retrieval-Augmented Generation (RAG) chat system for NASA mission documents covering Apollo 11, Apollo 13 and Challenger (STS-51-L). The system has four parts:

1. **Embedding pipeline** (`embedding_pipeline.py`): reads the 12 text files in `data_text/`, splits them into overlapping character chunks, embeds each chunk with OpenAI `text-embedding-3-small` and stores it in a persistent ChromaDB collection with metadata.
2. **Retrieval** (`rag_client.py`): embeds the user's question with the same model, runs a cosine-similarity search with a configurable top-k and an optional mission filter, removes duplicates, sorts by score and formats the excerpts as numbered, attributed context.
3. **Generation** (`llm_client.py`): a NASA mission expert system prompt, the retrieved context and the recent conversation history are sent to an OpenAI chat model, which answers with `[Source N]` citations and says when the documents do not contain the answer.
4. **Evaluation** (`ragas_evaluator.py`, `batch_evaluation.py`, `chat.py`): every answer in the Streamlit chat is scored in real time with RAGAS, and the questions in `evaluation_dataset.txt` can be evaluated in one batch with per-question and aggregate results.

The full end-to-end run (installation, database build, statistics, retrieval check, batch evaluation and chat test) was executed on Google Colab and is recorded in `run_end_to_end.ipynb`.

## 2. Design decisions

**Chunking.** Chunks are built in character windows of `--chunk-size` characters (default 500) with `--chunk-overlap` characters of overlap (default 100). Within each window the split point moves back to the nearest sentence end or line break, but only if that break lies in the last 40% of the window, so chunks stay close to the target size. The next chunk always starts exactly `chunk_overlap` characters before the end of the previous one. An offline check over all 12 files confirmed that no chunk exceeds 500 characters and that every consecutive pair overlaps by exactly 100 characters.

**Metadata.** Each chunk stores `source`, `file_path`, `mission` (apollo_11, apollo_13, challenger), `data_type`, `document_category`, `file_type`, `chunk_index`, `total_chunks`, character offsets and length. The mission field is used by the retrieval filter and every field used in source attributions comes from this metadata.

**Stable IDs and update modes.** Chunk IDs have the form `mission_source_chunk_0001`, so re-running the pipeline recognises existing chunks. `--update-mode skip` leaves them alone, `update` re-embeds and upserts them, and `replace` deletes all chunks of a source file before adding the new ones.

**Retrieval.** Questions are embedded explicitly with the model recorded in the collection metadata, so the question and the documents always share the same vector space. The client asks ChromaDB for k + 2 results, removes duplicate texts and keeps the best k by distance. The formatted context lists each excerpt as `[Source N] Mission | Document | Category`, separated by a divider line, and long excerpts are truncated.

**Prompting and history.** The system prompt instructs the model to act as a NASA mission operations expert, to base facts on the supplied excerpts, to cite sources, to flag garbled transcript text and to say clearly when the documents are insufficient. Only the last six user and assistant messages are passed as history, which keeps follow-up questions working without sending the whole conversation each time.

**Evaluation.** `evaluate_response_quality` validates its inputs (empty question or answer, error answers, missing or malformed contexts) and returns a clear error message instead of raising. Each metric is scored separately, so one failing metric does not hide the others. Response relevancy, faithfulness and context precision are always calculated. BLEU and ROUGE are added when a reference answer is supplied, as in batch evaluation.

## 3. Results

### 3.1 Vector database

| Item | Value |
|---|---|
| Source files | 12 (Apollo 11: 6, Apollo 13: 3, Challenger: 3) |
| Chunk size / overlap | 500 / 100 characters |
| Chunks stored | 16,512 (Apollo 11: 8,878, Apollo 13: 6,513, Challenger: 1,121) |
| Embedding model | text-embedding-3-small (cosine distance) |
| Re-run in skip mode | 0 added, 0 updated, 16,512 skipped |

### 3.2 Batch evaluation (`evaluation_dataset.txt`, k = 5, gpt-3.5-turbo)

| # | Category | Question | Relevancy | Faithfulness | Context precision | BLEU | ROUGE |
|---|---|---|---|---|---|---|---|
| 1 | overview | Main objective of Apollo 11 | 1.000 | 0.500 | 1.000 | 0.314 | 0.493 |
| 2 | emergency | Apollo 13 problem and consequences | 0.600 | 0.000 | 0.367 | 0.050 | 0.458 |
| 3 | disaster analysis | Challenger shortly after launch | 0.709 | 0.750 | 0.833 | 0.063 | 0.212 |
| 4 | crew | Apollo 13 crew members | 1.000 | 0.500 | 1.000 | 0.080 | 0.400 |
| 5 | technical | Apollo 13 carbon dioxide build-up | 1.000 | 1.000 | 1.000 | 0.102 | 0.232 |
| 6 | timeline | Apollo 11 landing and touchdown call | 0.792 | 0.000 | 1.000 | 0.098 | 0.542 |
| 7 | technical | Apollo 11 program alarms | 0.917 | 0.600 | 0.679 | 0.208 | 0.276 |

| Metric | Mean | Min | Max |
|---|---|---|---|
| Response relevancy | 0.860 | 0.600 | 1.000 |
| Context precision | 0.840 | 0.367 | 1.000 |
| Faithfulness | 0.479 | 0.000 | 1.000 |
| ROUGE | 0.373 | 0.212 | 0.542 |
| BLEU | 0.131 | 0.050 | 0.314 |

### 3.3 Interpretation

* **Relevancy and context precision are high.** The answers address the questions (mean 0.86) and most retrieved chunks are useful (mean 0.84). The mission filter keeps retrieval focused; for example, all top results for the Apollo 13 problem come from the Apollo 13 mission commentary.
* **Faithfulness is the weakest metric.** Two answers scored 0.0. In both cases the answer was factually correct but used background knowledge that was not in the retrieved excerpts: the Apollo 11 landing date (20 July 1969) and the "explosion" wording for Apollo 13, while the retrieved chunks only mention the loss of oxygen flow. RAGAS correctly flags these statements as unsupported by the context. This shows why the metric matters: a correct answer is not necessarily a grounded one.
* **BLEU and ROUGE are low by design.** They measure word overlap with the reference answers in `evaluation_dataset.txt`. The model words its answers differently and adds citations, so the scores are low even when the content agrees (for example, question 4 names exactly the same three crew members).

## 4. Challenges and solutions

| Challenge | Solution |
|---|---|
| RAGAS 0.4.3 fails to import because of old VertexAI import paths. | `fix_ragas_imports.py` applies the documented fix to the installed package and is safe to run repeatedly. |
| BLEU, ROUGE and the non-LLM context precision metric need extra packages. | Added `sacrebleu`, `rouge_score` and `rapidfuzz` to `requirements.txt`. |
| Udacity (Vocareum) keys need a different API endpoint. | `resolve_base_url` detects keys beginning with `voc-` and uses the Vocareum endpoint automatically; `OPENAI_BASE_URL` can still override it. |
| The Vocareum endpoint returns HTTP 429 (rate limit) during bulk embedding. | The OpenAI client retries automatically, and the pipeline embeds in batches of 100 with its own retry and back-off, so the full 16,512-chunk build completed with no errors. |
| ChromaDB 1.5.7 rejected question embeddings returned as numpy float32 values. | Retrieval converts the embedding to plain Python floats before querying. |
| The transcripts are OCR output with page markers, broken lines and speaker labels. | Chunks prefer sentence and line boundaries, and the system prompt tells the model to flag garbled text rather than guess. |
| Long conversations would send too many tokens. | Only the last six user and assistant turns are passed to the model. |

## 5. Possible improvements

* Tighten grounding: ask the model to answer only from the excerpts and to mark any background knowledge explicitly, which should raise faithfulness.
* Clean OCR artefacts (page markers, split words) before chunking.
* Add a re-ranking step or hybrid keyword search to improve context precision for broad questions such as question 2.
* Cache RAGAS scores for repeated questions to reduce evaluation time and cost.

## 6. How to reproduce

```bash
pip install -r requirements.txt
python fix_ragas_imports.py
export OPENAI_API_KEY="your-key"
python embedding_pipeline.py --data-path ./data_text --chunk-size 500 --chunk-overlap 100
python embedding_pipeline.py --stats-only
python batch_evaluation.py --dataset evaluation_dataset.txt --k 5
streamlit run chat.py
```
