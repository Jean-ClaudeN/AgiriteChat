"""
app.py — AgiriteChat Streamlit UI (styled version).

This preserves the original visual design (hero, metric cards, topbar, CSS)
and swaps the broken OpenAI + keyword-matching backend for the new
LangGraph agent + semantic retrieval + image analysis pipeline.

All business logic lives in: agent.py, retrieval.py, vision.py, llm.py, feedback.py
This file is UI only.
"""

import uuid
from html import escape

import streamlit as st

from agent import run as run_agent, get_retriever
from vision import analyze_field_image
from llm import is_available as llm_available
from feedback import log_interaction, record_feedback, recent_stats, init_db

# ---------------- Page config ----------------
st.set_page_config(
    page_title="AgiriteChat",
    layout="wide",
    page_icon="🌾",
)

# ---------------- One-time warm-up ----------------
@st.cache_resource
def _warm_up():
    """Load heavy dependencies once per session (embeddings model, DB)."""
    init_db()
    get_retriever()  # loads sentence-transformers + builds Chroma index
    return True

_warm_up()

# ---------------- Session state ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I can help with maize and soybean questions on pests, diseases, soil fertility, and fertilizer. What are you seeing in your field?",
        "structured": None,
        "top_score": 0.0,
        "needs_escalation": False,
        "interaction_id": None,
    }]

# ---------------- Original CSS (preserved from your app) ----------------
st.markdown("""
<style>
:root {
    --green-900: #1f4d2e;
    --green-800: #2f6b3d;
    --green-700: #3d7a49;
    --green-100: #eef6ef;
    --green-050: #f7fbf7;
    --text-main: #1f2937;
    --text-soft: #5b6470;
    --border-soft: #dfe7df;
    --card-bg: #ffffff;
    --shadow-soft: 0 10px 30px rgba(18, 38, 24, 0.08);
}
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
    color: var(--text-main);
}
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1240px;
}
.topbar {
    background: #ffffff;
    border: 1px solid #edf1ed;
    border-radius: 18px;
    padding: 0.95rem 1.2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1rem;
}
.brand {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--green-800);
    letter-spacing: 0.2px;
}
.nav-links {
    display: flex;
    gap: 1.4rem;
    color: var(--text-soft);
    font-size: 0.98rem;
    font-weight: 600;
}
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    min-height: 360px;
    padding: 2.2rem 2.3rem;
    background:
        linear-gradient(90deg, rgba(28,59,36,0.88) 0%, rgba(41,82,49,0.78) 40%, rgba(68,106,72,0.42) 100%),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.2rem;
}
.hero-content {
    max-width: 700px;
    color: #ffffff;
    padding-top: 1.2rem;
}
.hero-kicker {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.1rem;
    line-height: 1.02;
    font-weight: 750;
    margin-bottom: 0.9rem;
}
.hero-subtitle {
    font-size: 1.12rem;
    line-height: 1.55;
    color: rgba(255,255,255,0.92);
    margin-bottom: 1.25rem;
}
.cta-row {
    display: flex;
    gap: 0.9rem;
    flex-wrap: wrap;
}
.cta-note {
    margin-top: 0.8rem;
    color: rgba(255,255,255,0.85);
    font-size: 0.92rem;
}
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.9rem;
    margin: 1.1rem 0 1.4rem 0;
}
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border-soft);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    box-shadow: var(--shadow-soft);
}
.metric-label {
    color: var(--text-soft);
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.22rem;
    font-weight: 700;
    color: var(--text-main);
}
.section-card {
    background: var(--card-bg);
    border: 1px solid var(--border-soft);
    border-radius: 22px;
    padding: 1.2rem 1.25rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.28rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.section-subtitle {
    color: var(--text-soft);
    margin-bottom: 0.9rem;
}
.answer-card {
    background: var(--green-050);
    border: 1px solid #d8e6d7;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-top: 0.6rem;
    margin-bottom: 0.7rem;
}
.answer-label {
    font-weight: 700;
    color: var(--green-900);
    margin-bottom: 0.15rem;
}
.answer-block {
    margin-bottom: 0.85rem;
    line-height: 1.55;
}
.status-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: #edf6ed;
    color: var(--green-900);
    border: 1px solid #d9e9d8;
    font-size: 0.82rem;
    font-weight: 700;
    margin-right: 0.45rem;
    margin-bottom: 0.45rem;
}
.confidence-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-left: 0.4rem;
    vertical-align: middle;
}
.conf-high { background: #dcf1dc; color: #1f5d2e; border: 1px solid #b8dcb8; }
.conf-med  { background: #fff4d6; color: #7a5a00; border: 1px solid #ecd98f; }
.conf-low  { background: #fddede; color: #8a1f1f; border: 1px solid #f3b8b8; }
.escalate-box {
    background: #fff8e6;
    border-left: 4px solid #d9a41a;
    padding: 0.75rem 0.95rem;
    border-radius: 10px;
    margin-top: 0.8rem;
    font-size: 0.92rem;
    color: #5c4200;
}
.footer-box {
    background: #fafcf9;
    border: 1px solid #e4ece4;
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-top: 1rem;
    color: var(--text-soft);
    font-size: 0.92rem;
}
@media (max-width: 900px) {
    .metrics-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .hero-title {
        font-size: 2.2rem;
    }
    .nav-links {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- Rendering helpers ----------------
def render_answer_card(response: dict, top_score: float = 0.0, needs_escalation: bool = False):
    """Render the 5-section structured response with confidence badge."""
    if top_score >= 0.55:
        badge_class, badge_text = "conf-high", f"High confidence · {top_score:.2f}"
    elif top_score >= 0.35:
        badge_class, badge_text = "conf-med", f"Medium confidence · {top_score:.2f}"
    else:
        badge_class, badge_text = "conf-low", f"Low confidence · {top_score:.2f}"

    def safe(key):
        return escape(response.get(key, "") or "Not available")

    html = f"""
    <div class="answer-card">
        <div style="margin-bottom:0.5rem;">
            <span class="confidence-badge {badge_class}">{badge_text}</span>
        </div>
        <div class="answer-block">
            <div class="answer-label">Likely issue</div>
            <div>{safe("Likely issue")}</div>
        </div>
        <div class="answer-block">
            <div class="answer-label">Why this may be happening</div>
            <div>{safe("Why this may be happening")}</div>
        </div>
        <div class="answer-block">
            <div class="answer-label">What to check next</div>
            <div>{safe("What to check next")}</div>
        </div>
        <div class="answer-block">
            <div class="answer-label">Suggested action</div>
            <div>{safe("Suggested action")}</div>
        </div>
        <div class="answer-block" style="margin-bottom:0;">
            <div class="answer-label">When to seek local support</div>
            <div>{safe("When to seek local support")}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    if needs_escalation:
        st.markdown(
            '<div class="escalate-box">⚠️ <strong>Limited confidence in this answer.</strong> '
            'Please confirm with your local extension officer before taking action.</div>',
            unsafe_allow_html=True,
        )


def process_question(user_question: str, crop_hint: str, category_hint: str,
                     image_symptoms=None, image_source="none"):
    """Run the agent, render the result, log interaction, show feedback buttons."""
    with st.spinner("Thinking..."):
        state = run_agent(
            user_question=user_question,
            crop_hint=crop_hint,
            category_hint=category_hint,
            image_symptoms=image_symptoms or [],
            image_source=image_source,
        )

    interaction_id = log_interaction(st.session_state.session_id, state)

    render_answer_card(
        state.get("response", {}),
        state.get("top_score", 0.0),
        state.get("needs_escalation", False),
    )

    # Feedback
    fb1, fb2, _ = st.columns([1, 1, 10])
    if fb1.button("👍 Helpful", key=f"up_{interaction_id}"):
        record_feedback(interaction_id, 1)
        st.toast("Thanks for the feedback!")
    if fb2.button("👎 Not helpful", key=f"down_{interaction_id}"):
        record_feedback(interaction_id, -1)
        st.toast("Thanks — this helps us improve.")

    # Sources used
    matches = state.get("matches", [])
    if matches:
        with st.expander(f"Sources used ({len(matches)})"):
            for m in matches:
                st.markdown(
                    f"**{m['question']}**  \n"
                    f"*{m['crop']} · {m['category']} · match score {m['score']:.2f}*"
                )
                st.write(m["answer"])
                st.write("---")

    # Debug trace
    with st.expander("Agent trace (debug)"):
        st.write("**Path:**", " → ".join(state.get("trace", [])))
        st.write("**Classified crop:**", state.get("classified_crop", "unknown"))
        st.write("**Classified category:**", state.get("classified_category", "general"))
        st.write("**Top retrieval score:**", round(state.get("top_score", 0.0), 3))
        if state.get("image_source") and state.get("image_source") != "none":
            st.write("**Image analysis source:**", state.get("image_source"))

    # Persist in chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": state.get("response", {}).get("Likely issue", ""),
        "structured": state.get("response"),
        "top_score": state.get("top_score", 0.0),
        "needs_escalation": state.get("needs_escalation", False),
        "interaction_id": interaction_id,
    })


# ---------------- Topbar ----------------
st.markdown("""
<div class="topbar">
    <div class="brand">AgiriteChat</div>
    <div class="nav-links">
        <span>Who We Are</span>
        <span>How It Works</span>
        <span>Impact</span>
        <span>Knowledge Hub</span>
        <span>Field Support</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-kicker">AI assistant for agriculture</div>
        <div class="hero-title">Localized crop support for maize and soybean farmers</div>
        <div class="hero-subtitle">
            Helping farmers access timely, practical, and context-aware support for pests,
            diseases, soil fertility, fertilizer, and field decisions.
        </div>
        <div class="cta-row">
            <span class="status-pill">Localized support</span>
            <span class="status-pill">Crop-aware retrieval</span>
            <span class="status-pill">Structured guidance</span>
        </div>
        <div class="cta-note">
            Designed to support scalable farmer guidance beyond traditional one-to-one extension workflows.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("AgiriteChat")
    st.caption("Farmer support system for maize and soybean production")

    selected_crop = st.selectbox("Crop", ["General", "Maize", "Soybean"], index=0)
    selected_topic = st.selectbox(
        "Problem area",
        ["General", "Pest", "Disease", "Nutrient_deficiency", "Soil", "Weeds", "Drought", "Fertilizer"],
        index=0,
    )

    st.markdown("### System status")
    if llm_available():
        st.write("✅ AI reasoning: Active (Gemini)")
    else:
        st.write("⚠️ AI reasoning: Offline — set `GEMINI_API_KEY`")
    st.write("🔎 Semantic retrieval: Active")
    st.write("📸 Image analysis: Local + Gemini fallback")
    st.write("💾 Feedback logging: Active")

    with st.expander("Feedback stats"):
        stats = recent_stats()
        st.write(f"Total interactions: **{stats['total']}**")
        st.write(f"👍 {stats['thumbs_up']}   👎 {stats['thumbs_down']}")
        st.write(f"Escalations flagged: **{stats['escalations']}**")

    with st.expander("About AgiriteChat"):
        st.write(
            "AgiriteChat is an agent-based AI assistant for maize and soybean farmers. "
            "It uses a LangGraph state machine to classify questions, retrieve grounded "
            "knowledge with semantic search, check confidence, and ask clarifying "
            "questions when needed."
        )
        st.write(
            "Image analysis runs a local PlantVillage model for maize diseases and "
            "falls back to Gemini Vision for soybean and edge cases."
        )

# ---------------- Metric cards ----------------
ai_status_label = "AI + Knowledge Base" if llm_available() else "Knowledge Base Only"
st.markdown(f"""
<div class="metrics-grid">
    <div class="metric-card">
        <div class="metric-label">Supported crops</div>
        <div class="metric-value">Maize and Soybean</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">System mode</div>
        <div class="metric-value">{ai_status_label}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Current focus</div>
        <div class="metric-value">{escape(selected_topic)}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Consultation type</div>
        <div class="metric-value">Text and Photo Support</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Ask AgiriteChat", "Photo Review", "Knowledge Hub"])

# ---- Tab 1: Ask ----
with tab1:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Ask AgiriteChat</div>
        <div class="section-subtitle">
            Enter a crop question and receive structured support grounded in maize and soybean knowledge.
        </div>
    """, unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    if q1.button("Soybeans not fixing nitrogen"):
        st.session_state["preset_question"] = "My soybeans are weak and not fixing nitrogen well. What could be wrong?"
    if q2.button("Purple maize leaves"):
        st.session_state["preset_question"] = "My maize leaves are turning purple. What could be wrong?"
    if q3.button("Maize leaf blight"):
        st.session_state["preset_question"] = "How do I manage maize leaf blight?"
    if q4.button("Soybean root rot"):
        st.session_state["preset_question"] = "What causes root rot in soybean?"

    # Render chat history (structured rendering for past answers)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("structured"):
                render_answer_card(
                    message["structured"],
                    message.get("top_score", 0.0),
                    message.get("needs_escalation", False),
                )
            else:
                st.write(message["content"])

    # Handle preset click
    preset_question = st.session_state.pop("preset_question", None)
    if preset_question:
        st.session_state.messages.append({
            "role": "user",
            "content": preset_question,
            "structured": None,
        })
        with st.chat_message("user"):
            st.write(preset_question)
        with st.chat_message("assistant"):
            process_question(preset_question, selected_crop, selected_topic)

    # Handle typed input
    user_question = st.chat_input("Type your farming question here")
    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question,
            "structured": None,
        })
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            process_question(user_question, selected_crop, selected_topic)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Tab 2: Photo Review ----
with tab2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Photo Review</div>
        <div class="section-subtitle">
            Upload a field photo. The system will check image quality, analyze symptoms visually,
            and combine the findings with the knowledge base for structured guidance.
        </div>
    """, unsafe_allow_html=True)

    photo = st.file_uploader("Upload a field photo", type=["png", "jpg", "jpeg"])
    photo_description = st.text_input(
        "Optional description",
        placeholder="e.g. yellow patches on lower leaves, V6 stage"
    )

    if photo is not None:
        st.image(photo, use_container_width=True)

    if st.button("Analyze photo", type="primary"):
        if photo is None:
            st.warning("Please upload a photo first.")
        else:
            image_bytes = photo.getvalue()
            crop_hint_lower = selected_crop.lower() if selected_crop != "General" else None

            with st.spinner("Checking image quality and analyzing..."):
                vision_result = analyze_field_image(
                    image_bytes,
                    farmer_description=photo_description or "",
                    crop_hint=crop_hint_lower,
                )

            if not vision_result["ok"]:
                st.error(vision_result["quality_reason"] or "Could not analyze the image.")
            else:
                source_label = {
                    "plantvillage_local": "Local PlantVillage model",
                    "gemini": "Gemini Vision",
                }.get(vision_result["source"], vision_result["source"])
                st.success(f"Image analyzed via: **{source_label}**")

                if vision_result["symptoms"]:
                    st.markdown("**Symptoms detected:**")
                    for s in vision_result["symptoms"]:
                        st.write(f"• {s}")

                st.markdown("### Photo-based support")
                combined_q = photo_description or "Please help me diagnose this crop issue based on the photo."
                process_question(
                    user_question=combined_q,
                    crop_hint=selected_crop,
                    category_hint=selected_topic,
                    image_symptoms=vision_result["symptoms"],
                    image_source=vision_result["source"],
                )

                st.caption(
                    "Confidence note: this is advisory support based on visual description "
                    "and available knowledge, not a final diagnosis."
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Tab 3: Knowledge Hub ----
with tab3:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">Knowledge Hub</div>
        <div class="section-subtitle">
            Search the agronomic issues currently supported by the system, using semantic search.
        </div>
    """, unsafe_allow_html=True)

    search_term = st.text_input(
        "Search knowledge",
        placeholder="Search by crop, issue, deficiency, disease, pest, or fertilizer"
    )
    crop_filter = st.selectbox("Filter by crop", ["All", "Maize", "Soybean"], key="kh_crop_filter")

    if search_term:
        retriever = get_retriever()
        crop_arg = crop_filter.lower() if crop_filter != "All" else None
        hits = retriever.search(search_term, crop=crop_arg, top_k=10)
        st.write(f"**Results: {len(hits)}**")
        for h in hits:
            with st.expander(f"{h['question']}  ·  {h['crop']}/{h['category']}  ·  score {h['score']:.2f}"):
                st.write(h["answer"])
    else:
        # Browse mode (no search term)
        import json
        try:
            with open("knowledge_base.json", "r", encoding="utf-8") as f:
                entries = json.load(f)
        except FileNotFoundError:
            entries = []

        if crop_filter != "All":
            entries = [e for e in entries if e.get("crop") in (crop_filter.lower(), "both")]

        st.write(f"**Results: {len(entries)}**")
        for e in entries:
            with st.expander(f"{e['question']}  ·  {e.get('crop','?')}/{e.get('category','?')}"):
                st.write(e["answer"])
                if e.get("symptoms"):
                    st.caption("Symptoms: " + ", ".join(e["symptoms"]))

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="footer-box">
    <strong>AgiriteChat</strong><br>
    Version: 2.0 (agent-based)<br>
    Built as a crop-support system for maize and soybean production using LangGraph agent reasoning,
    semantic retrieval, image analysis, and feedback logging.<br><br>
    <strong>Responsible use:</strong> This tool is for support and early interpretation only.
    Serious disease, pest, and fertility issues should always be confirmed through local agronomic expertise.
</div>
""", unsafe_allow_html=True)
