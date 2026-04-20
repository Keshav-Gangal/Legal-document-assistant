"""
capstone_streamlit.py — Legal Document Assistant UI
Agentic AI Course 2026 | Dr. Kanthi Kiran Sirra
Run: streamlit run capstone_streamlit.py
"""

import uuid
import streamlit as st
from agent import initialise, ask

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD RESOURCES (cached — runs only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_agent():
    return initialise()   # returns (app, embedder, collection)

app, embedder, collection = load_agent()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {"role": "user"|"assistant", "content": str}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal Assistant")
   
    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}…`")

    st.markdown("**Domain:** Legal Document Assistant")
    st.markdown("**User:** Paralegals & Junior Lawyers")
    st.divider()

    st.markdown("### 📚 Knowledge Base Topics")
    topics = [
        "Contract Law — Essential Elements",
        "Contract Law — Breach & Remedies",
        "Contract Law — Vitiating Factors",
        "Criminal Procedure — Arrest Rights",
        "Criminal Procedure — Trial Process",
        "Civil Procedure — Filing a Suit",
        "Civil Procedure — Discovery & Evidence",
        "Tort Law — Negligence",
        "Tort Law — Strict & Vicarious Liability",
        "Constitutional — Fundamental Rights",
        "Intellectual Property — Copyright & Trademark",
        "Evidence Law — Admissibility & Burden of Proof",
    ]
    for t in topics:
        st.markdown(f"- {t}")
    st.divider()
    st.markdown("### 🛠️ Tools Available")
    st.markdown("- 📅 Current date & time\n- 🌐 Web search (DuckDuckGo)")




# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.title("⚖️ Legal Document Assistant")
st.caption("Ask questions about contract law, criminal procedure, civil procedure, torts, constitutional rights, IP, and evidence law.")

# ── Chat history ──────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].caption(f"🗺️ Route: `{meta.get('route', '—')}`")
            cols[1].caption(f"📊 Faithfulness: `{meta.get('faithfulness', '—')}`")
            if meta.get("sources"):
                cols[2].caption(f"📖 Sources: {', '.join(meta['sources'][:2])}")

# ── Chat input ────────────────────────────────
if prompt := st.chat_input("Ask a legal question…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = ask(app, prompt, thread_id=st.session_state.thread_id)

        answer = result.get("answer", "I'm sorry, I couldn't generate a response.")
        st.markdown(answer)

        # Show metadata
        meta = {
            "route":       result.get("route", "—"),
            "faithfulness": f"{result.get('faithfulness', 0):.2f}",
            "sources":     result.get("sources", []),
        }
        cols = st.columns(3)
        cols[0].caption(f"🗺️ Route: `{meta['route']}`")
        cols[1].caption(f"📊 Faithfulness: `{meta['faithfulness']}`")
        if meta["sources"]:
            cols[2].caption(f"📖 Sources: {', '.join(meta['sources'][:2])}")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "meta":    meta,
    })
