"""
YouTube Video Chatbot
----------------------
A Streamlit app that lets you paste a YouTube video link, builds a RAG
pipeline over its transcript, and lets you chat with the video.

Optionally evaluates each answer using RAGAS (faithfulness + answer
relevancy) so you can see how well-grounded the model's response is in
the retrieved transcript context.

Run with:
    streamlit run app.py
"""

import os
import re
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# TEMPORARY: SSL verification bypass for local dev only.
# Set DISABLE_SSL_VERIFY=true in your .env to enable. Do NOT enable this in
# any deployed/public version of the app -- it removes protection against
# man-in-the-middle attacks. This exists only to unblock local development
# on networks/machines with broken certificate chains.
# ---------------------------------------------------------------------------
if os.getenv("DISABLE_SSL_VERIFY", "false").lower() == "true":
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _original_request = requests.Session.request

    def _unverified_request(self, *args, **kwargs):
        kwargs["verify"] = False
        return _original_request(self, *args, **kwargs)

    requests.Session.request = _unverified_request

    # openai's SDK uses httpx, not requests -- needs its own bypass
    import httpx

    UNVERIFIED_HTTP_CLIENT = httpx.Client(verify=False)
else:
    UNVERIFIED_HTTP_CLIENT = None

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Supports either env var name so existing .env files keep working
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API")

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="🎥", layout="wide")

PROMPT = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=["context", "question"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_video_id(url_or_id: str) -> Optional[str]:
    """Pulls an 11-character YouTube video ID out of a full URL, or accepts
    a bare ID directly."""
    url_or_id = url_or_id.strip()
    pattern = r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/v\/|\/e\/|&v=)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    return None


@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    return " ".join(snippet.text for snippet in fetched)


@st.cache_resource(show_spinner=False)
def build_vectorstore(video_id: str, transcript: str, api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        http_client=UNVERIFIED_HTTP_CLIENT,
    )
    return FAISS.from_documents(chunks, embeddings)


def get_answer(question: str, retriever, llm, api_key: str):
    """Retrieves context, runs the LLM, and returns (answer, contexts)."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)
    final_prompt = PROMPT.invoke({"context": context_text, "question": question})
    answer = llm.invoke(final_prompt).content
    return answer, contexts


def evaluate_with_ragas(question: str, answer: str, contexts: List[str], api_key: str):
    """Scores an answer for faithfulness (is it grounded in the retrieved
    context?) and answer relevancy (does it actually address the question?)
    using RAGAS. Returns a dict of scores, or None if evaluation fails."""
    try:
        # RAGAS reads OPENAI_API_KEY from the environment internally
        os.environ["OPENAI_API_KEY"] = api_key

        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        dataset = Dataset.from_dict(
            {"question": [question], "answer": [answer], "contexts": [contexts]}
        )
        result = ragas_evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        row = result.to_pandas().iloc[0]
        return {
            "faithfulness": round(float(row["faithfulness"]), 2),
            "answer_relevancy": round(float(row["answer_relevancy"]), 2),
        }
    except Exception as e:  # noqa: BLE001
        st.warning(f"RAGAS evaluation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: question, answer, contexts, eval


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎥 YouTube Chatbot")

    if not API_KEY:
        st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env file.")

    video_input = st.text_input("YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")
    run_eval = st.checkbox("Evaluate answers with RAGAS", value=False,
                            help="Scores each answer for faithfulness and relevancy. Uses extra API calls.")
    load_clicked = st.button("Load video", type="primary", use_container_width=True)

    if load_clicked and video_input and API_KEY:
        video_id = extract_video_id(video_input)
        if not video_id:
            st.error("Couldn't parse a video ID from that input.")
        else:
            with st.spinner("Fetching transcript..."):
                transcript = None
                try:
                    transcript = fetch_transcript(video_id)
                except TranscriptsDisabled:
                    st.error("Captions are disabled for this video.")
                except NoTranscriptFound:
                    st.error("No English transcript found for this video.")
                except VideoUnavailable:
                    st.error("This video is unavailable.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Couldn't fetch transcript: {e}")

            if transcript:
                with st.spinner("Building knowledge base..."):
                    st.session_state.vector_store = build_vectorstore(video_id, transcript, API_KEY)
                st.session_state.video_id = video_id
                st.session_state.chat_history = []
                st.success("Video loaded. Ask away!")

    if st.session_state.video_id:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("Chat with a YouTube video")

if not st.session_state.vector_store:
    st.info("Paste a YouTube link in the sidebar and click **Load video** to get started.")
else:
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=API_KEY,
        http_client=UNVERIFIED_HTTP_CLIENT,
    )

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("eval"):
                c1, c2 = st.columns(2)
                c1.metric("Faithfulness", turn["eval"]["faithfulness"])
                c2.metric("Answer relevancy", turn["eval"]["answer_relevancy"])
            with st.expander("Sources used"):
                for i, ctx in enumerate(turn["contexts"], 1):
                    st.markdown(f"**Chunk {i}:** {ctx}")

    question = st.chat_input("Ask something about the video...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, contexts = get_answer(question, retriever, llm, API_KEY)
            st.write(answer)

            eval_scores = None
            if run_eval:
                with st.spinner("Scoring answer with RAGAS..."):
                    eval_scores = evaluate_with_ragas(question, answer, contexts, API_KEY)
                if eval_scores:
                    c1, c2 = st.columns(2)
                    c1.metric("Faithfulness", eval_scores["faithfulness"])
                    c2.metric("Answer relevancy", eval_scores["answer_relevancy"])

            with st.expander("Sources used"):
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(f"**Chunk {i}:** {ctx}")

        st.session_state.chat_history.append(
            {"question": question, "answer": answer, "contexts": contexts, "eval": eval_scores}
        )