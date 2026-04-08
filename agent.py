# agent.py

# ---------------------------------------------------------
# 1. LANGUAGES DICTIONARY (This is what caused the crash!)
# ---------------------------------------------------------
LANGUAGES = {
    "en": {"name": "English"},
    "sw": {"name": "Swahili"},
    "fr": {"name": "French"},
    "rw": {"name": "Kinyarwanda"}
}

# ---------------------------------------------------------
# 2. RETRIEVER SETUP
# ---------------------------------------------------------
def get_retriever():
    """
    Initializes and returns the document retriever.
    (Insert your actual VectorStore or FAISS/Chroma logic here)
    """
    class MockRetriever:
        def search(self, query, crop=None, top_k=3):
            return [] # Returns empty mock results for now
            
    return MockRetriever()

# ---------------------------------------------------------
# 3. MAIN AGENT RUN FUNCTION
# ---------------------------------------------------------
def run(user_question, crop_hint="General", category_hint="general", 
        image_symptoms=None, image_source="none", language="en", farmer_profile=None):
    """
    The main agent logic. app.py imports this as `run_agent`.
    (Replace the mock response below with your actual Groq/LangGraph code)
    """
    
    # Safely handle the farmer profile if provided
    profile = farmer_profile or {}
    name = profile.get("name", "Farmer")
    
    # The UI expects the response to be a dictionary with these exact keys:
    response_data = {
        "Likely issue": f"Evaluating: {user_question}",
        "Why this may be happening": f"Based on your {crop_hint} crop, this could be related to environmental stress or a common pathogen.",
        "What to check next": "Inspect the stems and the underside of the leaves for further symptoms.",
        "Suggested action": "Ensure proper irrigation and consider consulting your local agronomic guide.",
        "When to seek local support": "If symptoms spread rapidly to other plants within 48 hours."
    }
    
    # The UI expects the final state to look exactly like this:
    return {
        "response": response_data,
        "top_score": 0.75,            # Controls the "Confidence" badge (High/Med/Low)
        "needs_escalation": False,    # If True, shows the yellow warning box
        "matches": [],                # Sources from the knowledge base
        "trace": ["Input Received", "Intent Classified", "LLM Generated"],
        "language": language,
        "classified_crop": crop_hint,
        "image_source": image_source
    }
