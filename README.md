# AgiriteChat

**Agent-Based AI Decision Support for Sustainable Agriculture**

A LangGraph-powered crop advisory system for smallholder maize and soybean farmers — grounded in knowledge, personalized to the farm, available in 4 languages.

**Live prototype:** [agiritechat.streamlit.app](https://agiritechat.streamlit.app)

---

## Problem

Small-scale farmers in sub-Saharan Africa and across the global Corn Belt often lack timely access to expert advice for diagnosing crop threats. Extension officers are overstretched — one officer may serve 3,000+ farmers. By the time advice arrives, crops may already be lost. Existing AI tools function as general-purpose chatbots and may generate responses that are not grounded in domain-specific knowledge, raising concerns about reliability when applied to real farming conditions.

## Solution

AgiriteChat is an agent-based AI assistant that provides structured, context-aware guidance on crop problems through a multi-step agentic workflow. Instead of generating unchecked responses, the system classifies every question, retrieves from a curated knowledge base using semantic search, checks its own confidence before answering, and refuses to guess when uncertain.

**Core philosophy:** An agent is defined by knowing when not to act.

## Architecture

### LangGraph State Machine (7 Nodes)

```
User Input
    │
    ▼
┌─────────┐
│ Classify │──── greeting/off-topic ──→ [Greet] → friendly response
└────┬────┘
     │ agricultural question
     ▼
┌──────────┐
│ Retrieve │ semantic search (sentence-transformers + cosine similarity)
└────┬─────┘
     │ top-4 KB matches with scores
     ▼
┌──────────────────┐
│ Check Confidence │
└──┬────┬────┬─────┘
   │    │    │
   │    │    └── score < 0.35 ──→ [Refuse] → "I don't know, consult extension"
   │    └── crop name only ──→ [Clarify] → "What symptoms are you seeing?"
   └── score ≥ 0.35 ──→ [Synthesize] → grounded 5-section advisory
```

### Confidence Gating

| Score | Route | Behavior |
|-------|-------|----------|
| ≥ 0.55 | Synthesize | Full advisory with high confidence |
| 0.35–0.55 | Synthesize | Advisory with medium confidence + escalation note |
| < 0.35 | Refuse | Honest refusal, directs to extension officer |
| N/A | Clarify | Asks for symptoms when input is just a crop name |

### Source Separation

Every advisory response clearly labels what came from the verified knowledge base vs. what the LLM added:

- **"From knowledge base"** (green badge) — grounded in curated, source-attributed entries
- **"Additional AI guidance"** (amber badge) — LLM-generated context, explicitly labeled as unverified

## Features

- **Agent-based reasoning** — LangGraph state machine with 7 nodes, not a single prompt
- **Semantic retrieval** — sentence-transformers (all-MiniLM-L6-v2) + NumPy cosine similarity
- **Confidence gating** — explicit thresholds (0.55/0.35) with refusal path
- **Photo diagnosis** — Groq Vision (Llama 4 Scout 17B) extracts symptoms, KB provides diagnosis
- **Voice input** — Groq Whisper transcription for low-literacy users
- **Weather display** — Open-Meteo integration showing current conditions and 3-day forecast
- **4 languages** — English, Kiswahili, Français, Kinyarwanda (beta)
- **Farmer profiles** — session-based personalization (region, crops, farm size, planting date)
- **101-entry knowledge base** — curated for Nebraska and Rwanda Eastern Province with source attribution
- **Feedback logging** — SQLite with thumbs up/down for continuous improvement
- **Greeting detection** — non-agricultural inputs get conversational responses, not fake advisories
- **Structured answers** — 5-section format: Likely Issue, Why, What to Check, Action, When to Seek Support

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Text reasoning | Llama 3.3 70B via Groq |
| Vision analysis | Llama 4 Scout 17B via Groq |
| Voice transcription | Whisper Large v3 Turbo via Groq |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, local) |
| Agent framework | LangGraph (langgraph + langchain-core) |
| Retrieval | NumPy cosine similarity |
| Weather | Open-Meteo API (free, no key) |
| Knowledge base | 101 curated JSON entries |
| Feedback | SQLite |
| UI | Streamlit |
| Hosting | Streamlit Cloud (free tier) |
| Version control | GitHub |

## File Structure

```
AgiriteChat/
├── app.py              # Streamlit UI (2,100 lines) — landing page, tabs, rendering
├── agent.py            # LangGraph state machine (620 lines) — 7-node pipeline
├── llm.py              # Groq SDK wrapper — text, vision, Whisper
├── retrieval.py        # Semantic search — sentence-transformers + NumPy
├── vision.py           # Photo analysis — OpenCV quality check + Groq Vision
├── weather.py          # Open-Meteo weather display
├── feedback.py         # SQLite interaction logging
├── knowledge_base.json # 101 curated entries with source attribution
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Knowledge Base

101 entries covering maize and soybean production for **Nebraska** and **Rwanda Eastern Province** (Kayonza, Ngoma, Kirehe, Nyagatare, Bugesera).

**Categories:** pests (18), diseases (19), nutrient deficiencies (9), soil/water (9), planting/agronomy (11), fertilizer (3), harvest (3), storage (3), weeds (3), varieties (4), general (19)

**Sources:** 98 unique sources including UNL Extension publications (EC and G-series), CIMMYT, IITA, Rwanda Agriculture Board (RAB), FAO, ICIPE, and MINAGRI.

## Responsible AI

AgiriteChat is designed with safety guardrails for agricultural advisory:

- **Never invents information** — synthesis is constrained to retrieved KB entries
- **Never provides pesticide or fertilizer doses** — always refers to local extension
- **Refuses when uncertain** — low-confidence queries trigger honest "I don't know" responses
- **Separates verified from generated** — source badges distinguish KB content from AI guidance
- **Recommends extension** — every advisory includes "When to seek local support"
- **Vision describes, doesn't diagnose** — photo analysis extracts symptoms only; KB provides diagnosis
- **Transparent confidence** — every answer shows its retrieval confidence score

## Regional Focus

### Nebraska
- Corn Belt production practices (UNL Extension recommendations)
- Nebraska-specific pests: western corn rootworm, European corn borer, corn earworm
- Nebraska-specific diseases: Goss's wilt, bacterial leaf streak
- Precision agriculture, irrigation management, cover cropping
- UNL diagnostic lab and extension office referrals

### Rwanda Eastern Province
- Semi-arid conditions (800-1000 mm/year rainfall)
- District-specific guidance for Kayonza, Ngoma, Kirehe, Nyagatare, Bugesera
- Season A (September-October) and Season B (February-March) planting calendars
- RAB variety recommendations and fertilizer guidelines
- Aflatoxin prevention (critical in warm climate)
- Extension service contact points and farmer cooperatives

## Setup

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Development
```bash
pip install -r requirements.txt
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
streamlit run app.py
```

### Deployment (Streamlit Cloud)
1. Fork this repository
2. Connect to Streamlit Cloud
3. Add `GROQ_API_KEY` in Streamlit Cloud Secrets
4. Deploy

## Future Work

- Expand knowledge base from 101 to 300+ entries
- Voice input optimization for low-bandwidth environments
- Weather-aware recommendations (adjust advice based on forecast)
- React Native mobile app with offline caching
- Fine-tuned crop disease classifier on real farmer photos
- Real farmer testing with extension officers
- Partnership with agricultural NGOs

## Course

AGST 492: Agentic AI for Workflow Automation — University of Nebraska-Lincoln

## Author

Jean-Claude Niyomugabo

## License
---
This project is for educational purposes as part of AGST 492 at the University of Nebraska-Lincoln.
## Final Submission Materials

### 📄 Documents
- Final Report (PDF): [AgiriteChat_Final_Report .pdf](./AgiriteChat_Final_Report%20.pdf)  
- Presentation: [AgiriteChat_Presentation Document .pdf](./AgiriteChat_Presentation%20Document%20.pdf)  

### 🎥 Demo Video
This video demonstrates the system architecture, AI workflow, retrieval process, and a real agricultural use case.

- Watch here:  
  https://uofnebraska-my.sharepoint.com/:v:/r/personal/39808684_nebraska_edu/Documents/AgiriteChat_Presentation%20%20Video%20.mp4  

- Video link file:  
  [AgiriteChat_Video_Link.txt](./AgiriteChat_Video_Link.txt)  

### 🌐 Live Application
https://agiritechat.streamlit.app
