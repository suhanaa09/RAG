import streamlit as st
import os
import re
import time
import requests
from rag_engine import RAGEngine


# ── Tavily fallback helper ────────────────────────────────────────────────────
_LOW_CONFIDENCE_PATTERNS = [
    r"i (don'?t|do not) (have|find|know)",
    r"(not|no) (information|data|details|content|context|knowledge)",
    r"(cannot|can'?t|unable to) (find|answer|provide|help)",
    r"(outside|beyond) (my|the) (knowledge|scope|context|training)",
    r"(not (covered|available|found|indexed|in my))",
    r"i('m| am) (not sure|unsure|uncertain)",
    r"(no relevant|insufficient|limited) (context|information|data)",
    r"based on (the )?(available |provided )?context.{0,40}(cannot|can'?t|not able)",
    r"(this topic is not|i lack)",
]

def _is_low_confidence(answer: str) -> bool:
    """Return True when the RAG answer signals it couldn't find relevant info."""
    lower = answer.lower()
    for pat in _LOW_CONFIDENCE_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def _tavily_search(query: str, api_key: str, max_results: int = 5) -> dict:
    """
    Call Tavily Search API and return structured results.
    Wraps the call in a 60-second timeout — fits well within the 'same minute' requirement.
    """
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": False,
        "max_results": max_results,
        "topic": "general",
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=55)  # stay within 60 s
    resp.raise_for_status()
    data = resp.json()
    elapsed = round(time.time() - t0, 1)

    answer   = data.get("answer", "")
    results  = data.get("results", [])
    sources  = [r.get("url", "") for r in results if r.get("url")]
    snippets = [r.get("content", "") for r in results if r.get("content")]

    # Build a clean synthesis
    combined = answer if answer else " ".join(snippets[:3])
    return {
        "answer": combined.strip(),
        "sources": sources,
        "elapsed": elapsed,
    }

st.set_page_config(
    page_title="MediAssist AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #a78bfa !important; }
[data-testid="stChatMessage"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 12px !important; margin-bottom: 10px;
}
[data-testid="stChatInputTextArea"] {
    background: #161b27 !important; color: #e2e8f0 !important;
    border: 1px solid #7c3aed !important; border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: 13px !important;
    padding: 8px 16px !important; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #161b27 !important; color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important; border-radius: 8px !important;
}
.streamlit-expanderHeader {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 8px !important; color: #a78bfa !important;
    font-family: 'Space Mono', monospace !important; font-size: 12px !important;
}
.streamlit-expanderContent {
    background: #0d1117 !important; border: 1px solid #1e2330 !important;
    border-radius: 0 0 8px 8px !important;
}
.source-chip {
    display: inline-block; background: #1e2330; border: 1px solid #7c3aed44;
    color: #a78bfa; padding: 4px 10px; border-radius: 20px;
    font-size: 11px; font-family: 'Space Mono', monospace; margin: 3px;
}
.pubmed-chip {
    display: inline-block; background: #0d2137; border: 1px solid #3b82f644;
    color: #60a5fa; padding: 4px 10px; border-radius: 20px;
    font-size: 11px; font-family: 'Space Mono', monospace; margin: 3px;
}
.badge-success { background:#064e3b;color:#34d399;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-web     { background:#1e3050;color:#60a5fa;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-pubmed  { background:#0d2137;color:#38bdf8;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-warn    { background:#3b2000;color:#fbbf24;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-danger  { background:#3b0000;color:#f87171;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
.badge-scraper { background:#1a2e1a;color:#86efac;padding:3px 10px;border-radius:20px;font-size:12px;font-family:'Space Mono',monospace; }
[data-testid="metric-container"] {
    background: #161b27 !important; border: 1px solid #1e2330 !important;
    border-radius: 10px !important; padding: 12px !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #161b27 !important; border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #7c3aed55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
hr { border-color: #1e2330 !important; }
.stAlert { border-radius: 8px !important; }
.disclaimer {
    background: #1a0a0a; border: 1px solid #7f1d1d;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;
    color: #fca5a5; font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources_loaded" not in st.session_state:
        st.session_state.sources_loaded = []
    if "groq_key_set" not in st.session_state:
        st.session_state.groq_key_set = False

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🩺 MediAssist AI")
    st.markdown("*Medical RAG Assistant*")
    st.divider()

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 API Keys")

    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.text_input(
        "Tavily API Key (Live Web Search)",
        type="password",
        placeholder="tvly-...",
        help="Free at app.tavily.com — enables real-time web search fallback",
    )

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key  # stored for fallback helper
        if not st.session_state.groq_key_set:
            st.session_state.groq_key_set = True
            st.session_state.rag = RAGEngine(
                groq_api_key=groq_key,
                tavily_api_key=tavily_key,
            )
        if tavily_key and st.session_state.rag:
            st.session_state.rag.set_tavily_key(tavily_key)

        col1, col2 = st.columns(2)
        col1.markdown('<span class="badge-success">✓ Groq</span>', unsafe_allow_html=True)
        if tavily_key:
            col2.markdown('<span class="badge-web">✓ Web</span>', unsafe_allow_html=True)
        else:
            col2.markdown('<span class="badge-warn">⚠ No Web</span>', unsafe_allow_html=True)
    else:
        st.info("⚠️ Add your Groq API key to start")

    st.markdown(
        "<small style='color:#64748b'>PubMed search is always active — no key needed.<br>"
        "Tavily free tier: 1,000 searches/month.</small>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── RAG Tools status ──────────────────────────────────────────────────────
    st.markdown("### 🛠️ Active RAG Tools")
    st.markdown('<span class="badge-success">✓ FAISS+BM25+RRF</span>', unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown('<span class="badge-success">✓ Built-in Medical KB</span>', unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown('<span class="badge-success">✓ BM25 Keyword Search</span>', unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown('<span class="badge-success">✓ RRF Fusion Ranking</span>', unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown('<span class="badge-pubmed">✓ PubMed Literature</span>', unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown('<span class="badge-scraper">✓ Web Scraper Fallback</span>', unsafe_allow_html=True)
    st.markdown(" ")
    if st.session_state.rag and st.session_state.rag.live_search:
        st.markdown('<span class="badge-web">✓ Live Web Search</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠ Web Search (add Tavily)</span>', unsafe_allow_html=True)

    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Groq LLM",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    )
    if st.session_state.rag:
        st.session_state.rag.model = model

    st.divider()

    # ── Knowledge sources ─────────────────────────────────────────────────────
    st.markdown("### 📥 Add Medical Knowledge")
    tab_url, tab_text = st.tabs(["🌐 Web URL", "📝 Raw Text"])

    with tab_url:
        url_input = st.text_input("Medical URL", placeholder="https://mayoclinic.org/...")
        crawl_depth = st.slider("Crawl depth", 1, 3, 1)
        if st.button("🕷️ Scrape & Index", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not url_input:
                st.error("Enter a URL!")
            else:
                with st.spinner(f"Scraping {url_input}..."):
                    try:
                        result = st.session_state.rag.add_url(url_input, depth=crawl_depth)
                        st.session_state.sources_loaded.append({"type": "url", "src": url_input})
                        st.success(f"✓ Indexed {result['chunks']} chunks from {result['pages']} page(s)")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_text:
        raw_text = st.text_area("Paste clinical notes / guidelines", height=150, placeholder="Paste medical text, guidelines, case notes...")
        text_label = st.text_input("Label (optional)", placeholder="e.g. WHO Guidelines 2024")
        if st.button("📄 Index Text", use_container_width=True):
            if not st.session_state.rag:
                st.error("Add Groq API key first!")
            elif not raw_text.strip():
                st.error("Enter some text!")
            else:
                with st.spinner("Indexing..."):
                    try:
                        result = st.session_state.rag.add_text(raw_text, label=text_label or "Clinical Notes")
                        st.session_state.sources_loaded.append({"type": "text", "src": text_label or "Clinical Notes"})
                        st.success(f"✓ Indexed {result['chunks']} chunks")
                    except Exception as e:
                        st.error(f"Error: {e}")

    if st.session_state.sources_loaded:
        st.divider()
        st.markdown("### 📚 Loaded Sources")
        for s in st.session_state.sources_loaded:
            icon = "🌐" if s["type"] == "url" else "📝"
            label = s["src"][:35] + "…" if len(s["src"]) > 35 else s["src"]
            st.markdown(f'<span class="source-chip">{icon} {label}</span>', unsafe_allow_html=True)

    st.divider()

    # ── RAG settings ──────────────────────────────────────────────────────────
    st.markdown("### 🎛️ RAG Settings")
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    if st.session_state.rag:
        st.session_state.rag.top_k = top_k
        st.session_state.rag.temperature = temperature

    if st.session_state.rag and st.session_state.rag.vector_store:
        st.divider()
        st.markdown("### 📊 Index Stats")
        stats = st.session_state.rag.get_stats()
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["chunks"])
        c2.metric("Sources", stats["sources"])

    st.divider()
    if st.button("🗑️ Clear Everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_loaded = []
        if st.session_state.rag:
            st.session_state.rag.clear()
        st.success("Cleared!")
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("# 🩺 MediAssist AI")

# Disclaimer
st.markdown(
    '<div class="disclaimer">⚠️ <b>Medical Disclaimer:</b> This AI provides general medical information only. '
    'It is not a substitute for professional medical advice, diagnosis, or treatment. '
    'Always consult a qualified healthcare provider for medical decisions.</div>',
    unsafe_allow_html=True,
)

# Active tools badge row
has_web = st.session_state.rag and st.session_state.rag.live_search
badges = (
    '<span class="badge-success">✓ Built-in KB</span> &nbsp; '
    '<span class="badge-success">FAISS+BM25+RRF</span> &nbsp; '
    '<span class="badge-pubmed">📖 PubMed Active</span> &nbsp; '
    '<span class="badge-scraper">🕷️ Scraper Fallback ON</span>'
)
if has_web:
    badges += ' &nbsp; <span class="badge-web">🌐 Web Search ON</span>'
else:
    badges += ' &nbsp; <span class="badge-warn">⚠ No Tavily (Scraper active)</span>'
st.markdown(badges, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Badges
        col_badges = []
        if msg.get("used_fallback"):
            col_badges.append('<span class="badge-web">🌐 Auto Web Fallback</span>')
        elif msg.get("used_web_search"):
            col_badges.append('<span class="badge-web">🌐 Web Search</span>')
        if msg.get("used_scraper"):
            col_badges.append('<span class="badge-scraper">🕷️ Web Scraper</span>')
        if msg.get("used_pubmed"):
            col_badges.append('<span class="badge-pubmed">📖 PubMed</span>')
        if col_badges:
            st.markdown(" &nbsp; ".join(col_badges), unsafe_allow_html=True)
        # Sources
        all_sources = msg.get("sources", [])
        pubmed_sources = msg.get("pubmed_sources", [])
        doc_sources = [s for s in all_sources if s not in pubmed_sources]
        if pubmed_sources or doc_sources:
            with st.expander(f"📎 Sources ({len(all_sources)})"):
                for src in doc_sources:
                    st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)
                for src in pubmed_sources:
                    st.markdown(f'<span class="pubmed-chip">🔬 {src}</span>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a medical question — symptoms, conditions, treatments, medications…"):
    if not st.session_state.rag:
        st.warning("Please add your Groq API key in the sidebar first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge…"):
            try:
                result = st.session_state.rag.query(prompt)
                answer         = result["answer"]
                sources        = result.get("sources", [])
                pubmed_sources = result.get("pubmed_sources", [])
                used_web       = result.get("used_web_search", False)
                used_pubmed    = result.get("used_pubmed", False)
                used_scraper   = result.get("used_web_scraper", False)   # ← new
                used_fallback  = False   # auto web-search fallback flag
                fallback_elapsed = None

                # ── AUTO WEB FALLBACK ─────────────────────────────────────────
                # If the knowledge base couldn't answer AND Tavily key is set,
                # automatically fire a live web search (completes within ~60 s).
                tavily_key = os.environ.get("TAVILY_API_KEY", "")
                if _is_low_confidence(answer) and tavily_key:
                    st.toast("📡 KB had no answer — running live web search…", icon="🌐")
                    with st.spinner("🌐 Web fallback search in progress…"):
                        try:
                            wb = _tavily_search(prompt, tavily_key)
                            if wb["answer"]:
                                answer = (
                                    f"{wb['answer']}"
                                )
                                sources  = list(dict.fromkeys(sources + wb["sources"]))
                                used_web = True
                                used_fallback = True
                                fallback_elapsed = wb["elapsed"]
                        except requests.exceptions.Timeout:
                            st.warning("⏱️ Web search timed out (>60 s). Showing KB answer only.")
                        except Exception as web_err:
                            st.warning(f"⚠️ Web fallback failed: {web_err}")
                # ─────────────────────────────────────────────────────────────

                st.markdown(answer)

                col_badges = []
                if used_fallback:
                    elapsed_txt = f" ({fallback_elapsed}s)" if fallback_elapsed else ""
                    col_badges.append(
                        f'<span class="badge-web">🌐 Auto Web Fallback{elapsed_txt}</span>'
                    )
                elif used_web:
                    col_badges.append('<span class="badge-web">🌐 Web Search</span>')
                if used_scraper:
                    col_badges.append('<span class="badge-scraper">🕷️ Web Scraper</span>')
                if used_pubmed:
                    col_badges.append('<span class="badge-pubmed">📖 PubMed</span>')
                if col_badges:
                    st.markdown(" &nbsp; ".join(col_badges), unsafe_allow_html=True)

                doc_sources = [s for s in sources if s not in pubmed_sources]
                if pubmed_sources or doc_sources:
                    with st.expander(f"📎 Sources ({len(sources)})"):
                        for src in doc_sources:
                            st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)
                        for src in pubmed_sources:
                            st.markdown(f'<span class="pubmed-chip">🔬 {src}</span>', unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "pubmed_sources": pubmed_sources,
                    "used_web_search": used_web,
                    "used_pubmed": used_pubmed,
                    "used_fallback": used_fallback,
                    "used_scraper": used_scraper,
                })
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
