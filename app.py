# app.py
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import torch  # used to detect GPU availability

from utils import (
    extract_video_id,
    fetch_transcript,
    chunk_text,
    EmbedIndex,
    get_generator,
    answer_from_context,
)

st.set_page_config(page_title="YT Transcript Q&A", layout="wide")
st.title("YouTube Transcript Q&A (Streamlit)")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Retriever top-k (number of chunks)", min_value=1, max_value=10, value=4)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=200, step=50)
    use_gpu = st.checkbox("Use GPU for embeddings/LLM if available", value=False)
    st.markdown(
        "Tip: If you don't have a CUDA-capable GPU, leave this unchecked. "
        "Large generation models are slow on CPU; consider flan-t5-small for CPU."
    )

# Input: YouTube URL or ID
video_input = st.text_input("YouTube URL or Video ID", placeholder="https://www.youtube.com/watch?v=jG_RCGhYd1U")

if st.button("Load / Prepare"):
    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("Could not extract a YouTube video id from that input. Paste full URL or the ID.")
    else:
        st.session_state['video_id'] = video_id
        st.success(f"Video ID: {video_id}")

# Proceed if a video_id is set
if 'video_id' in st.session_state:
    video_id = st.session_state['video_id']
    st.subheader(f"Video ID: {video_id}")

    # Decide device string for EmbedIndex
    if use_gpu and torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    # instantiate embed index (lazy loads model when needed)
    embed_index = EmbedIndex(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        root_dir="data",
        device=device_str
    )

    # Load or build index
    if embed_index.index_exists(video_id):
        st.info("Found cached index for this video. Loading...")
        docs, _ = embed_index.load_index(video_id)
        st.write(f"Cached chunks: {len(docs)}")
        st.session_state['docs'] = docs
    else:
        st.info("No cache found: fetching transcript and building index (this may take a while).")
        try:
            with st.spinner("Fetching transcript..."):
                transcript = fetch_transcript(video_id, languages=["en"])
        except Exception as e:
            st.error(f"Error while fetching transcript: {type(e).__name__}: {e}")
            transcript = None

        if not transcript:
            st.error("Transcript not available for this video. Aborting processing.")
        else:
            st.success("Transcript fetched.")
            # chunk with UI settings
            docs = chunk_text(transcript, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            st.write(f"Chunks created: {len(docs)}")
            # build index (this will compute embeddings and save index)
            with st.spinner("Computing embeddings & building FAISS index..."):
                try:
                    index = embed_index.build_index(video_id, docs)
                except Exception as e:
                    st.error(f"Failed while building index: {type(e).__name__}: {e}")
                    index = None
            if index is not None:
                st.success("Index built and saved.")
                st.session_state['docs'] = docs

    st.markdown("---")
    st.subheader("Ask a question about the video")
    question = st.text_input("Your question", key="qa_input")

    if st.button("Ask"):
        if not question:
            st.warning("Type a question first.")
        else:
            docs = st.session_state.get('docs', [])
            if not docs:
                st.error("No docs loaded to answer from. Prepare the video first.")
            else:
                with st.spinner("Retrieving similar chunks..."):
                    try:
                        results = embed_index.query(video_id, question, top_k=int(top_k))
                    except Exception as e:
                        st.error(f"Retrieval error: {type(e).__name__}: {e}")
                        results = []

                if not results:
                    st.error("No matching chunks found.")
                else:
                    # show retrieved chunks
                    context = "\n\n".join([r[0] for r in results])
                    st.markdown("**Retrieved chunks (top)**")
                    for i, (txt, dist) in enumerate(results, 1):
                        st.write(f"**#{i}** (score {dist:.3f}): {txt[:400]}{'...' if len(txt)>400 else ''}")

                    # Decide device for generator pipeline: transformers expects device int (-1 or 0)
                    gen_device = 0 if (use_gpu and torch.cuda.is_available()) else -1
                    try:
                        generator = get_generator(device=gen_device)
                    except Exception as e:
                        st.error(f"Failed to create generator: {type(e).__name__}: {e}")
                        generator = None

                    if generator is None:
                        st.error("No generator available to produce an answer.")
                    else:
                        with st.spinner("Generating answer from LLM..."):
                            try:
                                # enforce single paragraph 4-point format in utils.answer_from_context
                                answer = answer_from_context(generator, context, question, enforce_single_paragraph_4points=True)
                                st.markdown("### Answer")
                                st.write(answer)
                                st.markdown("---")
                                st.caption("Answer generated only from the retrieved transcript chunks. Formatted as a single paragraph with four inline points.")
                            except Exception as e:
                                st.error(f"Generation error: {type(e).__name__}: {e}")
