"""
agent.py — LangGraph state machine for AgiriteChat.

Flow:
    classify -> retrieve -> check_confidence -> (clarify | synthesize) -> escalate_check -> END

Key design choices:
- Separate classification from retrieval so we can ask clarifying questions
  BEFORE embedding a vague query.
- Confidence gate uses retrieval similarity scores. If top hit is below
  HIGH_CONFIDENCE, we either ask a clarifying question or warn the farmer
  that the answer is uncertain.
- Image analysis is a separate entry point (run_with_image) that merges
  described symptoms into the text query before routing through the same
  retrieval pipeline. This keeps vision and knowledge grounding separate.
- Escalation is explicit: any answer with low confidence or dangerous
  keywords (pesticide dose, severe outbreak) gets a mandatory escalation
  note pointing the farmer to local extension support.
"""

import logging
from typing import TypedDict, List, Optional, Dict, Any, Literal

from langgraph.graph import StateGraph, END

from llm import generate_json, generate_text, is_available as llm_available
from retrieval import Retriever

logger = logging.getLogger(__name__)

# Confidence thresholds on retrieval similarity (0-1, higher = better match)
HIGH_CONFIDENCE = 0.55   # Good enough to answer directly
LOW_CONFIDENCE = 0.35    # Below this, refuse or escalate

# Keywords that always trigger explicit escalation in the response
ESCALATION_KEYWORDS = [
    "pesticide dose", "herbicide dose", "fertilizer rate",
    "dying", "whole field", "spreading fast", "entire crop",
    "dose", "dosage", "how much to spray",
]


class AgentState(TypedDict, total=False):
    # Inputs
    user_question: str
    crop_hint: str              # "maize" | "soybean" | "general"
    category_hint: str          # "pest" | "disease" | ... | "general"
    image_symptoms: List[str]   # from vision.py, empty if no image
    image_source: str           # "plantvillage_local" | "gemini" | "none"

    # Classification output
    classified_crop: str
    classified_category: str
    is_clear: bool
    clarification_needed: Optional[str]

    # Retrieval output
    matches: List[Dict]
    top_score: float

    # Routing decision
    route: Literal["clarify", "synthesize", "refuse"]

    # Final response
    response: Dict[str, str]      # structured answer
    needs_escalation: bool
    trace: List[str]              # step-by-step log for debugging/feedback


# Singleton retriever — loaded once, reused across requests
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# -------- Node implementations --------

def node_classify(state: AgentState) -> AgentState:
    """
    Classify the question: which crop, which category, is it clear enough?
    Uses the LLM if available, falls back to keyword rules.
    """
    state.setdefault("trace", []).append("classify")
    question = state["user_question"]
    crop_hint = state.get("crop_hint", "general")

    # Merge image symptoms into the question text so classification and
    # downstream retrieval both see them.
    image_symptoms = state.get("image_symptoms", [])
    if image_symptoms:
        question = f"{question}\n\nVisible symptoms from photo: {', '.join(image_symptoms)}"
        state["user_question"] = question

    if llm_available():
        prompt = f"""Classify this farming question.

Farmer question: {question}
Farmer selected crop: {crop_hint}

Return JSON only:
{{
  "crop": "maize" | "soybean" | "both" | "unknown",
  "category": "pest" | "disease" | "nutrient_deficiency" | "soil" | "weeds" | "drought" | "fertilizer" | "agronomy" | "harvest" | "nodulation" | "general",
  "is_clear": true | false,
  "clarification": "one short follow-up question if not clear, else empty string"
}}

Rules:
- "is_clear" = false if the crop is unknown AND the farmer didn't say, OR if symptoms are vague ("something is wrong").
- If farmer selected a crop, trust it unless question clearly contradicts.
- Keep clarification under 15 words and actionable (ask for ONE thing).
"""
        result = generate_json(prompt)
        if result:
            state["classified_crop"] = result.get("crop", "unknown")
            state["classified_category"] = result.get("category", "general")
            state["is_clear"] = bool(result.get("is_clear", True))
            state["clarification_needed"] = result.get("clarification", "") or None
            return state

    # Fallback classification (rule-based)
    q_lower = question.lower()
    if crop_hint != "general":
        state["classified_crop"] = crop_hint
    elif "soybean" in q_lower or "soya" in q_lower:
        state["classified_crop"] = "soybean"
    elif "maize" in q_lower or "corn" in q_lower:
        state["classified_crop"] = "maize"
    else:
        state["classified_crop"] = "unknown"

    state["classified_category"] = state.get("category_hint", "general").lower()
    state["is_clear"] = state["classified_crop"] != "unknown" and len(question.split()) >= 4
    if not state["is_clear"]:
        state["clarification_needed"] = "Which crop — maize or soybean — and what symptoms are you seeing?"
    return state


def node_retrieve(state: AgentState) -> AgentState:
    """Semantic search with metadata filters."""
    state.setdefault("trace", []).append("retrieve")
    retriever = get_retriever()

    crop = state.get("classified_crop", "unknown")
    if crop == "unknown":
        crop = None  # no filter

    hits = retriever.search(
        query=state["user_question"],
        crop=crop,
        category=None,  # don't over-filter; let semantics do the work
        top_k=4,
    )
    state["matches"] = hits
    state["top_score"] = hits[0]["score"] if hits else 0.0
    return state


def node_check_confidence(state: AgentState) -> AgentState:
    """Decide the route based on retrieval confidence and clarity."""
    state.setdefault("trace", []).append("check_confidence")

    if not state.get("is_clear", True) and state.get("clarification_needed"):
        state["route"] = "clarify"
        return state

    top = state.get("top_score", 0.0)
    if top < LOW_CONFIDENCE:
        state["route"] = "refuse"
    else:
        state["route"] = "synthesize"
    return state


def node_clarify(state: AgentState) -> AgentState:
    """Return a clarifying question instead of an answer."""
    state.setdefault("trace", []).append("clarify")
    q = state.get("clarification_needed") or "Could you share more details — which crop and what symptoms?"
    state["response"] = {
        "Likely issue": "More information needed",
        "Why this may be happening": "The question doesn't yet have enough detail to give reliable guidance.",
        "What to check next": q,
        "Suggested action": "Please reply with the crop, the symptoms you see, and the growth stage if known.",
        "When to seek local support": "If the problem is urgent or spreading quickly, contact your local extension officer while gathering these details.",
    }
    state["needs_escalation"] = False
    return state


def node_refuse(state: AgentState) -> AgentState:
    """Low-confidence refusal — honest 'I don't know' with escalation."""
    state.setdefault("trace", []).append("refuse")
    state["response"] = {
        "Likely issue": "Unable to give a confident answer",
        "Why this may be happening": "The question does not closely match any entry in my knowledge base. I would rather say I don't know than guess on a crop decision.",
        "What to check next": "Take photos of the affected plants, note the growth stage, and record how many plants or what fraction of the field is affected.",
        "Suggested action": "Please consult your local agricultural extension officer or a trusted agronomist before taking action.",
        "When to seek local support": "Now — this situation needs someone who can see the field directly.",
    }
    state["needs_escalation"] = True
    return state


def node_synthesize(state: AgentState) -> AgentState:
    """Generate the structured answer, grounded in retrieved matches."""
    state.setdefault("trace", []).append("synthesize")
    matches = state["matches"]
    top_score = state["top_score"]

    context = "\n\n".join(
        f"[Source {i+1}] Q: {m['question']}\nA: {m['answer']}"
        for i, m in enumerate(matches)
    )

    # If we have an image that produced candidate conditions, mention them.
    image_context = ""
    if state.get("image_symptoms"):
        image_context = f"\n\nVisible symptoms from uploaded photo ({state.get('image_source', 'unknown')}): {', '.join(state['image_symptoms'])}"

    system = (
        "You are AgiriteChat, a practical agricultural advisor for smallholder "
        "maize and soybean farmers. You give clear, simple, field-ready guidance. "
        "You NEVER invent information not in the provided sources. "
        "You NEVER give specific pesticide or fertilizer doses — always refer to "
        "local extension for doses. You keep language simple and concrete."
    )

    prompt = f"""Farmer's question: {state['user_question']}{image_context}

Retrieved knowledge base sources:
{context}

Using ONLY the information in the sources above, answer in this exact JSON format:
{{
  "Likely issue": "one short phrase naming the most probable issue",
  "Why this may be happening": "2-3 plain sentences explaining the likely cause",
  "What to check next": "2-3 specific things the farmer can check in the field today",
  "Suggested action": "2-3 concrete actions, no specific chemical doses",
  "When to seek local support": "specific conditions that should trigger a call to extension"
}}

Rules:
- If the sources don't support a confident answer, say so in "Likely issue".
- Do not invent diseases, pests, or treatments not in the sources.
- Never give specific pesticide or fertilizer doses.
- Match the crop the farmer asked about.
"""

    result = generate_json(prompt, system=system, temperature=0.2)
    if result and all(k in result for k in ["Likely issue", "Why this may be happening", "What to check next", "Suggested action", "When to seek local support"]):
        state["response"] = result
    else:
        # LLM failed or returned bad shape — fall back to top match verbatim.
        top = matches[0]
        state["response"] = {
            "Likely issue": top["question"].rstrip("?"),
            "Why this may be happening": top["answer"],
            "What to check next": "Compare the described symptoms with what you see in the field, and check growth stage.",
            "Suggested action": "Use this guidance as a starting point and confirm with direct field observation.",
            "When to seek local support": "If symptoms are spreading or affecting a large area, contact your local extension officer.",
        }
        state.setdefault("trace", []).append("synthesize_fallback")

    # Escalation check
    needs_esc = top_score < HIGH_CONFIDENCE
    q_lower = state["user_question"].lower()
    if any(kw in q_lower for kw in ESCALATION_KEYWORDS):
        needs_esc = True
    state["needs_escalation"] = needs_esc

    if needs_esc:
        # Strengthen the escalation line
        current = state["response"].get("When to seek local support", "")
        state["response"]["When to seek local support"] = (
            "Contact your local extension officer to confirm this diagnosis before acting. "
            + current
        )

    return state


# -------- Graph construction --------

def _route_from_confidence(state: AgentState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", node_classify)
    g.add_node("retrieve", node_retrieve)
    g.add_node("check_confidence", node_check_confidence)
    g.add_node("clarify", node_clarify)
    g.add_node("refuse", node_refuse)
    g.add_node("synthesize", node_synthesize)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "check_confidence")
    g.add_conditional_edges(
        "check_confidence",
        _route_from_confidence,
        {
            "clarify": "clarify",
            "refuse": "refuse",
            "synthesize": "synthesize",
        },
    )
    g.add_edge("clarify", END)
    g.add_edge("refuse", END)
    g.add_edge("synthesize", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# -------- Public entry points --------

def run(
    user_question: str,
    crop_hint: str = "general",
    category_hint: str = "general",
    image_symptoms: Optional[List[str]] = None,
    image_source: str = "none",
) -> Dict[str, Any]:
    """Main entry point. Returns full state dict for logging + display."""
    graph = get_graph()
    initial: AgentState = {
        "user_question": user_question,
        "crop_hint": crop_hint.lower(),
        "category_hint": category_hint.lower(),
        "image_symptoms": image_symptoms or [],
        "image_source": image_source,
        "trace": [],
    }
    final_state = graph.invoke(initial)
    return final_state
