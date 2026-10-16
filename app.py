"""Streamlit web interface for PureLogics RAG Assistant."""

import os
import streamlit as st

# ============ PAGE CONFIG (Sab se pehla Streamlit command) ============
st.set_page_config(
    page_title="PureLogics RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============ HEADER ============
st.title("🤖 PureLogics RAG Assistant")
st.markdown(
    "Ask questions about **PureLogics** — answers come only from verified public data."
)
st.markdown("---")

# ============ SIDEBAR ============
with st.sidebar:
    st.header("⚙️ Settings")

    # API key: Streamlit Cloud secrets ya manual input
    api_key = os.environ.get("GROQ_API_KEY", "")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = st.text_input(
                "Groq API Key:",
                type="password",
                help="Get a free key at console.groq.com",
            )

    st.markdown("---")
    st.markdown("### 📚 Data Sources")
    st.markdown(
        """
        - About PureLogics
        - AI Services
        - Mobile App Development
        - Digital Commerce
        - Quality Assurance
        - Industries Served
        - ClosedWon Case Study
        - Careers
        """
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        """
        This RAG system:
        - Retrieves relevant info
        - Refuses to hallucinate
        - Shows source files
        """
    )

# ============ PIPELINE (Session State) ============
# Session state ensures pipeline loads only once per user session
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# Load pipeline when API key is available
if api_key and st.session_state.pipeline is None:
    with st.spinner("🔄 Loading RAG pipeline (first time only)..."):
        try:
            from src.pipeline import RAGPipeline
            st.session_state.pipeline = RAGPipeline(api_key)
            st.success(f"✅ Pipeline ready — {st.session_state.pipeline.num_chunks} chunks loaded")
        except Exception as e:
            st.error(f"❌ Failed to load pipeline: {e}")

# ============ MAIN INTERFACE ============
if not api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to start.")
    st.info("🔑 Get a free key at [console.groq.com](https://console.groq.com)")
    st.stop()

if st.session_state.pipeline is None:
    st.info("⏳ Pipeline is loading. Please wait...")
    st.stop()

# ============ QUERY INPUT ============
st.markdown("### 💬 Ask a Question")

# Example questions
st.markdown("**Try these examples:**")

example_cols = st.columns(3)
examples = [
    "When was PureLogics founded?",
    "What AI services does PureLogics offer?",
    "What is the salary of the CEO?",
]

for col, example in zip(example_cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.current_query = example

# Input box
query = st.text_input(
    "Your question:",
    value=st.session_state.get("current_query", ""),
    placeholder="e.g., What mobile services does PureLogics offer?",
    key="query_input",
)

# ============ HANDLE SUBMISSION ============
if st.button("🚀 Ask", type="primary", use_container_width=True):
    if not query or not query.strip():
        st.warning("⚠️ Please type a question first.")
    else:
        with st.spinner("🤔 Searching and thinking..."):
            try:
                result = st.session_state.pipeline.ask(query.strip())

                # Answer
                st.markdown("### ✅ Answer")
                if result["refused"]:
                    st.warning(result["answer"])
                else:
                    st.success(result["answer"])

                # Sources
                if result["sources"]:
                    st.markdown("### 📚 Sources")
                    for source in result["sources"]:
                        st.markdown(f"- `{source}`")

                # Optional: Show retrieved chunks (debug)
                with st.expander("🔍 View retrieved context (debug)"):
                    for i, r in enumerate(result["retrieved"], 1):
                        st.markdown(f"**Rank {i}** — Score: `{r['score']:.3f}` — Source: `{r['source']}`")
                        st.caption(r["chunk"])

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============ FOOTER ============
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.85em;">
    Built with ❤️ using Streamlit, sentence-transformers, and Groq LLM<br>
    <a href="https://github.com/Malik-Usman2324CyberMind/Purelogics-rag-assistant" target="_blank">View on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
