# YouTube Transcript Q\&A — README

A  Streamlit app that fetches YouTube transcripts, creates a FAISS vector index over transcript chunks using `sentence-transformers`, and answers user questions grounded in the transcript using an LLM generation pipeline (transformers). The app enforces a single-paragraph answer with **four inline points**: `1) ... 2) ... 3) ... 4)`.

---

## Features

* Accepts a YouTube URL or video ID and extracts the video ID automatically.
* Uses your working `YouTubeTranscriptApi().fetch(...)` logic to fetch transcripts.
* Splits transcripts into chunks (configurable size & overlap).
* Computes embeddings with `sentence-transformers` and persists a FAISS index under `data/{video_id}`.
* Retrieves top-k chunks, constructs context and asks an LLM to answer — output forced to a **single paragraph with four inline points**.
* Lazy-loads heavy models (embeddings) to avoid startup crashes.
* Simple caching: reuses saved index/docs if already processed.

---

## Repo structure

```
your_project/
├─ app.py                 # Streamlit app (entrypoint)
├─ utils.py               # helpers: youtube id extraction, transcript fetching, chunking, embeddings, FAISS save/load, generation formatting
├─ requirements.txt
├─ .env.example
├─ data/                  # created at runtime, contains per-video caches (index.faiss + docs.pkl)
├─ models/                # optional: predownloaded HF models (not required)
└─ README.md
```

---

## Requirements

Use the included `requirements.txt`. Example:

```
streamlit>=1.20.0
youtube-transcript-api>=0.6.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4.post2
transformers>=4.30.0
torch>=2.0.0
python-dotenv>=1.0.0
```

> On Windows, for a reliable CPU-only torch install run:
>
> ```bash
> pip uninstall -y torch torchvision torchaudio
> pip install --upgrade pip
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Environment variables

Copy `.env.example` → `.env` and fill in values as needed.

`.env.example`:

```
# Optional: if you use private HF models or the Hub
HUGGINGFACEHUB_API_TOKEN=your_token_here

# Optional flag to force GPU usage for generation/embeddings
# USE_GPU=1
```

**Important:** Never commit real tokens to version control.

---

## How to run (local)

1. Create & activate a virtualenv:

   * macOS / Linux:

     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   * Windows:

     ```powershell
     python -m venv venv
     venv\Scripts\activate
     ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` → `.env` (if needed) and adjust.

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501` in your browser.

---

## Usage

1. Paste a YouTube URL or ID into the input box — click **Load / Prepare**.
2. If a cached index for that video exists under `data/{video_id}` it will load it. Otherwise it will:

   * Fetch transcript using `YouTubeTranscriptApi().fetch(video_id, languages=["en"])` (with safe fallbacks).
   * Chunk the transcript, compute embeddings and build a FAISS index (saved to `data/{video_id}`).
3. Enter a question about the video and click **Ask**.
4. The app retrieves top-k chunks, builds a context, then asks the LLM to answer. The output is formatted as a single paragraph containing four inline numbered points (1) ... 4)).

---

## Configuration options (UI)

* **Retriever top-k**: number of chunks to retrieve.
* **Chunk size** / **Chunk overlap**: controls the text splitting behavior.
* **Use GPU**: if a CUDA-capable GPU is available and you check this, the app will attempt to use it. Otherwise, it falls back to CPU.

---

## Notes & troubleshooting

### Common error: `YouTubeTranscriptApi` attribute errors

The app uses the same `ytt_api = YouTubeTranscriptApi()` + `ytt_api.fetch(video_id, languages=["en"])` approach that worked in your Colab. If you see `get_transcript` errors, make sure your `utils.fetch_transcript` uses `fetch(...)` not `get_transcript(...)`.

### Common error: `SentenceTransformer` / device / NotImplementedError

* We lazy-load `SentenceTransformer` and pass an explicit `device` string (e.g. `'cpu'` or `'cuda:0'`). This avoids the model being moved to a bad device at import time.
* If you installed a CUDA torch build but don’t have the correct GPU/drivers, prefer a CPU-only PyTorch build. See Windows CPU install above.

### Model speed & memory

* `google/flan-t5-large` is large and can be slow or memory-heavy on CPU. For faster local responses use `google/flan-t5-small` in `utils.get_generator()` if you’re CPU-bound.
* If you plan production deployment, consider using a remote inference endpoint (Hugging Face Inference API, Replicate, or other) and call that from the app.

### Caching / reprocessing

* Created FAISS indexes and docs are stored under `data/{video_id}`. If you want to reprocess a video, delete that folder for the video and re-run **Load / Prepare**.

### Transcript not available

* If transcripts/captions are disabled by the uploader or not available in the requested language, the app will show an informative error. You can attempt other languages or manually upload transcripts as a future enhancement.

---

## Development tips

* To speed iteration, set `GEN_MODEL = "google/flan-t5-small"` in `utils.py` while developing on CPU.
* To precompute indexes: run a small script that calls `fetch_transcript()` + `chunk_text()` + `embed_index.build_index()` for a list of video IDs — this prevents end-users from waiting while the index is built.
* Add authentication to your Streamlit app if you plan to deploy publicly.

---

## Contributing

1. Fork the repo.
2. Create a branch for your change.
3. Open a PR describing the change and why it helps.

---

## License

MIT © MUHAMMAD HASSAN SHAHBAZ
---

## Acknowledgements

* `youtube-transcript-api` for transcript retrieval
* `sentence-transformers` for embeddings
* `faiss` for vector similarity
* `transformers` for generation pipelines
* Streamlit for the UI

