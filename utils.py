# utils.py
import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import numpy as np

# ---- Configurable defaults ----
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# ---- Utilities ----
def extract_video_id(url_or_id: str) -> Optional[str]:
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_\-]{8,}", s):
        return s
    m = re.search(r"(?:v=|\/)([A-Za-z0-9_\-]{8,})", s)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be\/([A-Za-z0-9_\-]{8,})", s)
    if m:
        return m.group(1)
    return None

def fetch_transcript(video_id: str, languages=["en"]) -> Optional[str]:
    ytt_api = YouTubeTranscriptApi()
    try:
        segments = ytt_api.fetch(video_id, languages=languages)
        texts = []
        for s in segments:
            if hasattr(s, "text"):
                texts.append(s.text)
            elif isinstance(s, dict):
                texts.append(s.get("text", ""))
            else:
                texts.append(str(s))
        transcript = " ".join(t for t in texts if t).strip()
        return transcript or None
    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except Exception:
        raise

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        if end >= length:
            chunks.append(text[start:length].strip())
            break
        window = text[start:end]
        cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("! "), window.rfind("? "))
        if cut == -1 or cut < chunk_size // 2:
            cut = chunk_size
            chunk = text[start:start + cut].strip()
            start = start + cut - chunk_overlap
        else:
            chunk = text[start:start + cut + 1].strip()
            start = start + cut + 1 - chunk_overlap
        chunks.append(chunk)
    return [c for c in chunks if c]

# ---- Embeddings and FAISS wrappers ----
class EmbedIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 root_dir: str = "data", device: str = "cpu"):
        """
        device: 'cpu' or 'cuda:0' (or other valid torch device strings)
        NOTE: model is NOT loaded here. It will be loaded lazily when needed.
        """
        self.model_name = model_name
        self.root_dir = Path(root_dir)
        self._embedder = None
        self._dim = None
        self.device = device

    def _video_dir(self, video_id: str) -> Path:
        d = self.root_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def index_exists(self, video_id: str) -> bool:
        d = self._video_dir(video_id)
        return (d / "index.faiss").exists() and (d / "docs.pkl").exists()

    def _ensure_embedder(self):
        """Lazy-load the SentenceTransformer and set dimension."""
        if self._embedder is None:
            # If user asked for GPU but it's not available, fallback to CPU and warn
            if "cuda" in self.device and not torch.cuda.is_available():
                # fallback to CPU
                self.device = "cpu"
            # Instantiate with explicit device to avoid .to() surprises
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._embedder.get_sentence_embedding_dimension()

    @property
    def dim(self):
        if self._dim is None:
            # ensure embedder loaded to know dim
            self._ensure_embedder()
        return self._dim

    def save_index(self, video_id: str, docs: List[str], index: faiss.IndexFlatL2):
        d = self._video_dir(video_id)
        faiss.write_index(index, str(d / "index.faiss"))
        with open(d / "docs.pkl", "wb") as f:
            pickle.dump(docs, f)

    def load_index(self, video_id: str) -> Tuple[List[str], faiss.IndexFlatL2]:
        d = self._video_dir(video_id)
        index = faiss.read_index(str(d / "index.faiss"))
        with open(d / "docs.pkl", "rb") as f:
            docs = pickle.load(f)
        return docs, index

    def build_index(self, video_id: str, docs: List[str]) -> faiss.IndexFlatL2:
        # ensure model ready
        self._ensure_embedder()
        embeddings = self._embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        index = faiss.IndexFlatL2(self.dim)
        index.add(np.array(embeddings).astype("float32"))
        self.save_index(video_id, docs, index)
        return index

    def query(self, video_id: str, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
        if not self.index_exists(video_id):
            raise ValueError("Index does not exist for video_id")
        docs, index = self.load_index(video_id)
        self._ensure_embedder()
        q_emb = self._embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(docs):
                results.append((docs[idx], float(dist)))
        return results

# ---- LLM / generator + formatting ----
def get_generator(model_name: str = GEN_MODEL, device: int = -1):
    gen = pipeline(
        "text2text-generation",
        model=model_name,
        max_new_tokens=256,
        do_sample=False,
        device=device
    )
    return gen

def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def _force_four_inline_points_from_text(text: str) -> str:
    """
    Fallback: take up to 4 sentences from text and build a single-paragraph inline-numbered result.
    """
    text = _clean_whitespace(text)
    # split into sentences naively by punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text
    # take up to 4
    parts = sentences[:4]
    # if less than 4, try to split by semicolon or comma to get more parts
    if len(parts) < 4:
        extra = re.split(r';\s*|\s*-\s*|\s*,\s*', text)
        extra = [e.strip() for e in extra if e.strip()]
        # prefer existing parts then fill from extra
        parts = (parts + [e for e in extra if e not in parts])[:4]
    # format inline
    inline = " ".join(f"{i+1}) {parts[i]}" for i in range(len(parts)))
    return inline

def answer_from_context(generator, context: str, question: str, enforce_single_paragraph_4points: bool = True) -> str:
    """
    Generator prompt will instruct the model to respond in a single paragraph with four inline points.
    If the model doesn't, fallback logic will try to enforce the format.
    """
    # Strong instruction in the prompt about single-paragraph output
    prompt = (
        "You are a helpful assistant. Answer ONLY from the provided transcript context. "
        "If the context is insufficient, say 'I don't know'.\n\n"
        "IMPORTANT: Output must be a single paragraph (no line breaks) and must contain FOUR concise points "
        "inline in this exact style: 1) first point. 2) second point. 3) third point. 4) fourth point.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    out = generator(prompt, max_new_tokens=256)
    if isinstance(out, list) and out:
        raw = out[0].get("generated_text", out[0].get("text", ""))
    else:
        raw = str(out)

    cleaned = _clean_whitespace(raw).replace("\n", " ")
    # If already contains inline numbering 1) 2) 3) 4) return cleaned
    if enforce_single_paragraph_4points:
        if re.search(r'\b1\)\s*', cleaned) and re.search(r'\b2\)\s*', cleaned) and re.search(r'\b3\)\s*', cleaned) and re.search(r'\b4\)\s*', cleaned):
            # ensure single paragraph (no newlines) and normalized spaces
            return cleaned
        # fallback: attempt to construct from sentences
        fallback = _force_four_inline_points_from_text(cleaned)
        # ensure punctuation between items ends with a period
        # normalize so there is a period at end of each item if missing
        def ensure_period(s: str) -> str:
            s = s.strip()
            return s if s.endswith(('.', '!', '?')) else s + '.'
        # split fallback by pattern of '(digit))' to re-normalize periods
        parts = re.split(r'(?<=\))\s*', fallback)
        # if parts don't look correct just return fallback
        if len(parts) >= 1 and ')' in parts[0]:
            # clean up and rejoin as single paragraph with 1) ... 2) ...
            return _clean_whitespace(fallback)
        return _clean_whitespace(fallback)
    else:
        return cleaned
