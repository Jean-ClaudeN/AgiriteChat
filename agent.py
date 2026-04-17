"""
agent.py — LangGraph state machine for AgiriteChat.

Flow:
    classify -> route_or_retrieve -> (greet | retrieve -> check_confidence -> clarify | refuse | synthesize) -> END

v4 changes:
- Greeting/off-topic detection: "Hi", "thanks", non-agricultural questions get a
  conversational response instead of a fake advisory
- Source-separated answers: response now includes "kb_answer" (direct from knowledge base)
  and "ai_guidance" (LLM-generated additional context), clearly separated
- Language parameter: agent responds in English, Swahili, French, or Kinyarwanda
- Farmer profile context: region, farm size, crops, planting date
"""

import logging
import re
from typing import TypedDict, List, Optional, Dict, Any, Literal

from langgraph.graph import StateGraph, END

from llm import generate_json, generate_text, is_available as llm_available
from retrieval import Retriever

logger = logging.getLogger(__name__)

# Confidence thresholds on retrieval similarity (0-1, higher = better match)
HIGH_CONFIDENCE = 0.55
LOW_CONFIDENCE = 0.35

# Keywords that always trigger explicit escalation
ESCALATION_KEYWORDS = [
    "pesticide dose", "herbicide dose", "fertilizer rate",
    "dying", "whole field", "spreading fast", "entire crop",
    "dose", "dosage", "how much to spray",
]

# Patterns that indicate NON-agricultural input
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|yo|sup)\s*[!.,?]*\s*$",
    r"^\s*(thanks?|thank\s*you|thx|merci|asante|murakoze)\s*[!.,?]*\s*$",
    r"^\s*(bye|goodbye|see\s*you|take\s*care)\s*[!.,?]*\s*$",
    r"^\s*(ok|okay|yes|no|sure|cool|great|nice|wow)\s*[!.,?]*\s*$",
    r"^\s*(test|testing|123|abc)\s*[!.,?]*\s*$",
    r"^\s*(who\s*are\s*you|what\s*are\s*you|what\s*can\s*you\s*do)\s*[!.,?]*\s*$",
    r"^\s*(how\s*are\s*you|what'?s\s*up)\s*[!.,?]*\s*$",
]

# Conversational responses for different greeting types
GREETING_RESPONSES = {
    "en": {
        "greeting": "Hello! I'm AgiriteChat, your crop advisory assistant for maize and soybean farming. How can I help you today? You can ask me about pests, diseases, soil problems, or upload a photo of your crop.",
        "thanks": "You're welcome! If you have more questions about your crops, I'm here to help.",
        "farewell": "Goodbye! Remember, for serious crop problems, always confirm with your local extension officer. Good luck with your farming!",
        "who": "I'm AgiriteChat, an AI crop advisor for smallholder maize and soybean farmers. I can help you identify crop problems, suggest actions, and point you to local support when needed. Try asking me about a specific issue you're seeing in your field.",
        "other": "I'm designed to help with maize and soybean farming questions. Could you describe a crop problem you're experiencing? For example: 'My maize leaves are turning yellow' or 'I see holes in my soybean pods'.",
    },
    "sw": {
        "greeting": "Habari! Mimi ni AgiriteChat, msaidizi wako wa ushauri wa mazao kwa kilimo cha mahindi na soya. Nawezaje kukusaidia leo?",
        "thanks": "Karibu! Ukiwa na maswali zaidi kuhusu mazao yako, niko hapa kukusaidia.",
        "farewell": "Kwaheri! Kumbuka, kwa matatizo makubwa ya mazao, thibitisha kila mara na afisa ugani wako wa karibu.",
        "who": "Mimi ni AgiriteChat, mshauri wa mazao wa AI kwa wakulima wadogo wa mahindi na soya. Jaribu kuniuliza kuhusu tatizo maalum unaloliona shambani kwako.",
        "other": "Nimeundwa kusaidia na maswali ya kilimo cha mahindi na soya. Je, unaweza kuelezea tatizo la mazao unalokutana nalo?",
    },
    "fr": {
        "greeting": "Bonjour! Je suis AgiriteChat, votre assistant conseil pour la culture du maïs et du soja. Comment puis-je vous aider aujourd'hui?",
        "thanks": "De rien! Si vous avez d'autres questions sur vos cultures, je suis là pour vous aider.",
        "farewell": "Au revoir! N'oubliez pas, pour les problèmes graves, confirmez toujours avec votre agent de vulgarisation local.",
        "who": "Je suis AgiriteChat, un conseiller agricole IA pour les petits agriculteurs de maïs et de soja. Essayez de me poser une question sur un problème spécifique dans votre champ.",
        "other": "Je suis conçu pour aider avec les questions sur le maïs et le soja. Pouvez-vous décrire un problème de culture que vous rencontrez?",
    },
    "rw": {
        "greeting": "Muraho! Ndi AgiriteChat, umufasha wawe w'inama z'imyaka ku buhinzi bw'ibigori n'ibishyimbo. Nagufasha nte uyu munsi?",
        "thanks": "Murakaza neza! Niba ufite ibibazo byinshi ku bihingwa byawe, ndi hano kugufasha.",
        "farewell": "Murabeho! Ibuka, ku bibazo bikomeye by'ibihingwa, buri gihe wemeze n'umujyanama w'ubuhinzi wo hafi.",
        "who": "Ndi AgiriteChat, umujyanama w'ibihingwa wa AI ku bahinzi bato b'ibigori n'ibishyimbo. Gerageza kumbaza ikibazo kidasanzwe ubona mu murima wawe.",
        "other": "Nakoze kugira ngo mfashe ibibazo by'ubuhinzi bw'ibigori n'ibishyimbo. Ushobora gusobanura ikibazo cy'igihingwa uhura nacyo?",
    },
}


# Language config: name for UI, instruction for the LLM
LANGUAGES = {
    "en": {"name": "English", "instruction": "Respond in clear, simple English."},
    "sw": {"name": "Kiswahili", "instruction": "Jibu kwa Kiswahili sanifu, rahisi na cha wazi."},
    "fr": {"name": "Français", "instruction": "Réponds en français clair et simple, adapté aux agriculteurs."},
    "rw": {"name": "Kinyarwanda", "instruction": "Subiza mu Kinyarwanda cyoroshye kandi gisobanutse."},
}


class AgentState(TypedDict, total=False):
    # Inputs
    user_question: str
    crop_hint: str
    category_hint: str
    image_symptoms: List[str]
    image_source: str
    language: str                # "en" | "sw" | "fr" | "rw"
    farmer_profile: Dict[str, str]  # name, region, farm_size, crops, planting_date

    # Classification output
    classified_crop: str
    classified_category: str
    is_clear: bool
    is_agricultural: bool       # NEW: False for greetings/off-topic
    greeting_type: str          # NEW: "greeting" | "thanks" | "farewell" | "who" | "other"
    clarification_needed: Optional[str]

    # Retrieval output
    matches: List[Dict]
    top_score: float

    # Routing decision
    route: Literal["greet", "clarify", "synthesize", "refuse"]

    # Final response
    response: Dict[str, str]
    kb_sources: List[Dict]      # NEW: the actual KB entries used
    needs_escalation: bool
    trace: List[str]


_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _language_instruction(state: AgentState) -> str:
    lang = state.get("language", "en")
    return LANGUAGES.get(lang, LANGUAGES["en"])["instruction"]


def _farmer_context(state: AgentState) -> str:
    """Build a farmer context block to inject into prompts."""
    profile = state.get("farmer_profile") or {}
    if not profile or not any(profile.values()):
        return ""
    parts = []
    if profile.get("name"):
        parts.append(f"Farmer name: {profile['name']}")
    if profile.get("region"):
        parts.append(f"Region: {profile['region']}")
    if profile.get("farm_size"):
        parts.append(f"Farm size: {profile['farm_size']}")
    if profile.get("crops"):
        parts.append(f"Crops grown: {profile['crops']}")
    if profile.get("planting_date"):
        parts.append(f"Planting date: {profile['planting_date']}")
    if not parts:
        return ""
    return "Farmer context:\n" + "\n".join(f"- {p}" for p in parts) + "\n"


def _detect_greeting(text: str) -> Optional[str]:
    """Check if text is a greeting/non-agricultural message. Returns type or None."""
    text_clean = text.strip().lower()

    # Very short messages (1-3 words) that aren't crop-related are likely greetings
    if len(text_clean.split()) <= 2:
        for pattern in GREETING_PATTERNS:
            if re.match(pattern, text_clean, re.IGNORECASE):
                # Determine type
                if re.match(r".*(thank|thx|merci|asante|murakoze).*", text_clean):
                    return "thanks"
                if re.match(r".*(bye|goodbye|see\s*you|take\s*care).*", text_clean):
                    return "farewell"
                if re.match(r".*(who|what\s*are\s*you|what\s*can).*", text_clean):
                    return "who"
                return "greeting"

    # Check all patterns regardless of length
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, text_clean, re.IGNORECASE):
            if re.match(r".*(thank|thx|merci|asante|murakoze).*", text_clean):
                return "thanks"
            if re.match(r".*(bye|goodbye|see\s*you|take\s*care).*", text_clean):
                return "farewell"
            if re.match(r".*(who|what\s*are|what\s*can).*", text_clean):
                return "who"
            return "greeting"

    return None


# -------- Node implementations --------

def node_classify(state: AgentState) -> AgentState:
    state.setdefault("trace", []).append("classify")
    question = state["user_question"]
    crop_hint = state.get("crop_hint", "general")

    # Step 0: Check if this is a greeting or non-agricultural message
    greeting_type = _detect_greeting(question)
    if greeting_type:
        state["is_agricultural"] = False
        state["greeting_type"] = greeting_type
        state["classified_crop"] = "none"
        state["classified_category"] = "greeting"
        state["is_clear"] = True
        return state

    # If we get here, treat as potentially agricultural
    state["is_agricultural"] = True
    state["greeting_type"] = ""

    # Step 1: Check if input is too short or just a crop name with no problem
    # "maize", "soybean", "corn", "my maize", "soya" — these aren't questions
    q_stripped = question.strip().lower()
    q_words = q_stripped.split()
    just_crop_words = {"maize", "corn", "soybean", "soya", "soy", "beans", "crop", "crops", "plant", "plants", "field", "farm", "my"}

    if len(q_words) <= 3 and all(w.strip(".,!?") in just_crop_words for w in q_words):
        # It's just a crop name, not a question
        state["is_agricultural"] = True
        state["classified_crop"] = "maize" if any(w in q_stripped for w in ["maize", "corn"]) else "soybean" if any(w in q_stripped for w in ["soybean", "soya", "soy"]) else "unknown"
        state["classified_category"] = "general"
        state["is_clear"] = False
        state["clarification_needed"] = "What problem are you seeing with your crop? Describe the symptoms — for example, yellow leaves, holes, wilting, or spots."
        return state

    image_symptoms = state.get("image_symptoms", [])
    if image_symptoms:
        question = f"{question}\n\nVisible symptoms from photo: {', '.join(image_symptoms)}"
        state["user_question"] = question

    if llm_available():
        farmer_ctx = _farmer_context(state)
        prompt = f"""Classify this farming question.

{farmer_ctx}Farmer question: {question}
Farmer selected crop: {crop_hint}

Return JSON only:
{{
  "is_agricultural": true | false,
  "crop": "maize" | "soybean" | "both" | "unknown",
  "category": "pest" | "disease" | "nutrient_deficiency" | "soil" | "weeds" | "drought" | "fertilizer" | "agronomy" | "harvest" | "nodulation" | "storage" | "variety" | "general",
  "is_clear": true | false,
  "clarification": "one short follow-up question if not clear, else empty string"
}}

Rules:
- "is_agricultural" = false if the question is a greeting, off-topic, or has nothing to do with farming.
- "is_clear" = false if the crop is unknown AND the farmer didn't say, OR if symptoms are vague.
- If farmer selected a crop, trust it unless question clearly contradicts.
- Keep clarification under 15 words and actionable (ask for ONE thing).
- The clarification should be in English (internal use only).
"""
        result = generate_json(prompt)
        if result:
            # Check if LLM says it's not agricultural
            if not result.get("is_agricultural", True):
                state["is_agricultural"] = False
                state["greeting_type"] = "other"
                state["classified_crop"] = "none"
                state["classified_category"] = "off_topic"
                state["is_clear"] = True
                return state

            state["classified_crop"] = result.get("crop", "unknown")
            state["classified_category"] = result.get("category", "general")
            state["is_clear"] = bool(result.get("is_clear", True))
            state["clarification_needed"] = result.get("clarification", "") or None
            return state

    # Fallback classification
    q_lower = question.lower()

    # Simple non-agricultural check for fallback
    agri_keywords = [
        "maize", "corn", "soybean", "soya", "crop", "plant", "leaf", "leaves",
        "pest", "disease", "soil", "fertilizer", "weed", "harvest", "seed",
        "yellow", "brown", "wilting", "spots", "holes", "rot", "blight",
        "armyworm", "borer", "rust", "nodule", "nitrogen", "phosphorus",
        "field", "farm", "planting", "germination", "drought", "rain",
    ]
    has_agri_keyword = any(kw in q_lower for kw in agri_keywords)

    if not has_agri_keyword and len(question.split()) < 5:
        state["is_agricultural"] = False
        state["greeting_type"] = "other"
        state["classified_crop"] = "none"
        state["classified_category"] = "off_topic"
        state["is_clear"] = True
        return state

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


def node_greet(state: AgentState) -> AgentState:
    """Handle greetings, thank-yous, and off-topic messages conversationally."""
    state.setdefault("trace", []).append("greet")
    lang = state.get("language", "en")
    greeting_type = state.get("greeting_type", "greeting")

    responses = GREETING_RESPONSES.get(lang, GREETING_RESPONSES["en"])
    message = responses.get(greeting_type, responses["other"])

    state["response"] = {
        "type": "conversational",
        "message": message,
    }
    state["kb_sources"] = []
    state["needs_escalation"] = False
    return state


def node_retrieve(state: AgentState) -> AgentState:
    state.setdefault("trace", []).append("retrieve")
    retriever = get_retriever()

    crop = state.get("classified_crop", "unknown")
    if crop == "unknown":
        crop = None

    hits = retriever.search(
        query=state["user_question"],
        crop=crop,
        category=None,
        top_k=4,
    )
    state["matches"] = hits
    state["top_score"] = hits[0]["score"] if hits else 0.0
    return state


def node_check_confidence(state: AgentState) -> AgentState:
    state.setdefault("trace", []).append("check_confidence")

    top = state.get("top_score", 0.0)
    is_clear = state.get("is_clear", True)
    has_clarification = bool(state.get("clarification_needed"))

    # If the question was explicitly marked as unclear (e.g. just a crop name
    # with no symptoms), ALWAYS clarify — the KB match is noise, not signal.
    if not is_clear and has_clarification:
        state["route"] = "clarify"
        return state

    # Clear question with a good KB match → synthesize
    if top >= LOW_CONFIDENCE:
        state["route"] = "synthesize"
        return state

    # Low retrieval score, clear question → refuse honestly
    state["route"] = "refuse"
    return state


def node_clarify(state: AgentState) -> AgentState:
    """Return a clarifying question — translated into the chosen language."""
    state.setdefault("trace", []).append("clarify")
    q_en = state.get("clarification_needed") or "Could you share more details — which crop and what symptoms?"

    lang = state.get("language", "en")

    # For non-English, ask the LLM to translate the clarification naturally
    if lang != "en" and llm_available():
        trans_prompt = f"Translate this short farming clarification question into {LANGUAGES[lang]['name']}, keeping it friendly and simple. Return only the translation, nothing else.\n\nEnglish: {q_en}"
        translated = generate_text(trans_prompt, temperature=0.1)
        if translated:
            q_en = translated.strip()

    state["response"] = _translated_fallback({
        "type": "advisory",
        "Likely issue": "More information needed",
        "Why this may be happening": "The question doesn't yet have enough detail to give reliable guidance.",
        "What to check next": q_en,
        "Suggested action": "Please reply with the crop, the symptoms you see, and the growth stage if known.",
        "When to seek local support": "If the problem is urgent or spreading quickly, contact your local extension officer while gathering these details.",
    }, state)
    state["kb_sources"] = []
    state["needs_escalation"] = False
    return state


def node_refuse(state: AgentState) -> AgentState:
    """Honest 'I don't know' with escalation, translated if needed."""
    state.setdefault("trace", []).append("refuse")
    state["response"] = _translated_fallback({
        "type": "advisory",
        "Likely issue": "Unable to give a confident answer",
        "Why this may be happening": "The question does not closely match any entry in my knowledge base. I would rather say I don't know than guess on a crop decision.",
        "What to check next": "Take photos of the affected plants, note the growth stage, and record how many plants or what fraction of the field is affected.",
        "Suggested action": "Please consult your local agricultural extension officer or a trusted agronomist before taking action.",
        "When to seek local support": "Now — this situation needs someone who can see the field directly.",
    }, state)
    state["kb_sources"] = []
    state["needs_escalation"] = True
    return state


def _translated_fallback(response_en: Dict[str, str], state: AgentState) -> Dict[str, str]:
    """Translate a fixed English fallback response into the chosen language."""
    lang = state.get("language", "en")
    if lang == "en" or not llm_available():
        return response_en

    import json
    # Don't translate the "type" key
    to_translate = {k: v for k, v in response_en.items() if k != "type"}
    prompt = (
        f"Translate these agricultural advisory sections into {LANGUAGES[lang]['name']}. "
        "Keep the JSON structure exactly the same, only translate the values. "
        "Return ONLY the translated JSON, no markdown.\n\n"
        + json.dumps(to_translate, ensure_ascii=False)
    )
    translated = generate_json(prompt)
    if translated and all(k in translated for k in to_translate.keys()):
        translated["type"] = response_en.get("type", "advisory")
        return translated
    return response_en


def node_synthesize(state: AgentState) -> AgentState:
    """Generate the structured answer with source separation."""
    state.setdefault("trace", []).append("synthesize")
    matches = state["matches"]
    top_score = state["top_score"]

    # Store the KB sources for display
    state["kb_sources"] = [
        {
            "question": m["question"],
            "answer": m["answer"],
            "crop": m.get("crop", ""),
            "category": m.get("category", ""),
            "source": m.get("source", "Knowledge Base"),
            "score": m["score"],
        }
        for m in matches[:3]  # Top 3 sources
    ]

    context = "\n\n".join(
        f"[Source {i+1}: {m.get('source', 'Knowledge Base')}] Q: {m['question']}\nA: {m['answer']}"
        for i, m in enumerate(matches)
    )

    image_context = ""
    if state.get("image_symptoms"):
        image_context = f"\n\nVisible symptoms from uploaded photo ({state.get('image_source', 'unknown')}): {', '.join(state['image_symptoms'])}"

    farmer_ctx = _farmer_context(state)
    lang_instruction = _language_instruction(state)

    system = (
        "You are AgiriteChat, a practical agricultural advisor for smallholder "
        "maize and soybean farmers. You give clear, simple, field-ready guidance. "
        "You NEVER invent information not in the provided sources. "
        "You NEVER give specific pesticide or fertilizer doses — always refer to "
        "local extension for doses. You keep language simple and concrete. "
        f"{lang_instruction}"
    )

    prompt = f"""{farmer_ctx}Farmer's question: {state['user_question']}{image_context}

Retrieved knowledge base sources:
{context}

Using ONLY the information in the sources above, answer in this exact JSON format:
{{
  "Likely issue": "one short phrase naming the most probable issue",
  "Why this may be happening": "2-3 plain sentences explaining the likely cause, referencing the knowledge base sources",
  "What to check next": "2-3 specific things the farmer can check in the field today",
  "Suggested action": "2-3 concrete actions, no specific chemical doses",
  "When to seek local support": "specific conditions that should trigger a call to extension",
  "AI additional context": "1-2 sentences of helpful general context that goes BEYOND what the sources say. Clearly state this is general AI guidance, not from the verified knowledge base. If there is nothing useful to add, say 'No additional context needed.'"
}}

Rules:
- The first 5 sections MUST only use information from the knowledge base sources above.
- The "AI additional context" section is the ONLY place where you can add general knowledge.
- If the sources don't support a confident answer, say so in "Likely issue".
- Do not invent diseases, pests, or treatments not in the sources.
- Never give specific pesticide or fertilizer doses.
- Match the crop the farmer asked about.
- {lang_instruction}
- The JSON keys MUST stay in English, but the VALUES should be in the target language.
"""

    result = generate_json(prompt, system=system, temperature=0.2)
    expected_keys = ["Likely issue", "Why this may be happening", "What to check next", "Suggested action", "When to seek local support"]
    if result and all(k in result for k in expected_keys):
        result["type"] = "advisory"
        state["response"] = result
    else:
        # Fallback: use top match verbatim, translated
        top = matches[0]
        fallback_en = {
            "type": "advisory",
            "Likely issue": top["question"].rstrip("?"),
            "Why this may be happening": top["answer"],
            "What to check next": "Compare the described symptoms with what you see in the field, and check growth stage.",
            "Suggested action": "Use this guidance as a starting point and confirm with direct field observation.",
            "When to seek local support": "If symptoms are spreading or affecting a large area, contact your local extension officer.",
            "AI additional context": "No additional context needed.",
        }
        state["response"] = _translated_fallback(fallback_en, state)
        state.setdefault("trace", []).append("synthesize_fallback")

    # Escalation check
    needs_esc = top_score < HIGH_CONFIDENCE
    q_lower = state["user_question"].lower()
    if any(kw in q_lower for kw in ESCALATION_KEYWORDS):
        needs_esc = True
    state["needs_escalation"] = needs_esc

    return state


# -------- Graph construction --------

def _route_after_classify(state: AgentState) -> str:
    """Route based on whether the input is agricultural or not."""
    if not state.get("is_agricultural", True):
        return "greet"
    return "retrieve"


def _route_from_confidence(state: AgentState) -> str:
    return state["route"]


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", node_classify)
    g.add_node("greet", node_greet)
    g.add_node("retrieve", node_retrieve)
    g.add_node("check_confidence", node_check_confidence)
    g.add_node("clarify", node_clarify)
    g.add_node("refuse", node_refuse)
    g.add_node("synthesize", node_synthesize)

    g.set_entry_point("classify")

    # After classify: either greet (non-agricultural) or retrieve (agricultural)
    g.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "greet": "greet",
            "retrieve": "retrieve",
        },
    )

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
    g.add_edge("greet", END)
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


# -------- Public entry point --------

def run(
    user_question: str,
    crop_hint: str = "general",
    category_hint: str = "general",
    image_symptoms: Optional[List[str]] = None,
    image_source: str = "none",
    language: str = "en",
    farmer_profile: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Main entry point. Returns full state dict for logging + display."""
    graph = get_graph()
    initial: AgentState = {
        "user_question": user_question,
        "crop_hint": crop_hint.lower(),
        "category_hint": category_hint.lower(),
        "image_symptoms": image_symptoms or [],
        "image_source": image_source,
        "language": language if language in LANGUAGES else "en",
        "farmer_profile": farmer_profile or {},
        "trace": [],
    }
    final_state = graph.invoke(initial)
    return final_state
