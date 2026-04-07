# 🌾 AgiriteChat v2.0 — Agent-Based AI Crop Support

An agent-based AI assistant for maize and soybean farmers, built on LangGraph,
Gemini, semantic retrieval, and local image analysis.

This is **not** a simple chatbot. It follows a structured agent workflow:

```
classify → retrieve → check_confidence → (clarify | refuse | synthesize) → END
```

Every answer is grounded in a curated knowledge base. Low-confidence questions
trigger clarifying follow-ups or explicit escalation to local extension, instead
of guessing.

---

## ✨ What's new in v2.0

Compared to the original prototype:

| Aspect | v1.0 | v2.0 |
| --- | --- | --- |
| Retrieval | Keyword bag-of-words scoring | Semantic search (sentence-transformers + ChromaDB) |
| Reasoning | Linear pipeline | LangGraph agent with conditional routing |
| Confidence | None (always answered) | Confidence gate: clarify / refuse / answer |
| Image analysis | **Not implemented** (only description used) | Quality check + local PlantVillage model + Gemini Vision fallback |
| LLM | OpenAI (with a broken SDK call) | Google Gemini 2.0 Flash (free tier) |
| Grounding | Prompt instruction only | Structured sources, cited in UI, fallback rules |
| Feedback | None | SQLite logging + thumbs up/down per answer |
| KB structure | Flat Q&A | Enriched schema: crop, category, symptoms, growth stage, confidence |

---

## 📁 Project structure

```
AgiriteChat/
├── app.py                    # Streamlit UI (styled, wired to agent)
├── agent.py                  # LangGraph state machine
├── retrieval.py              # ChromaDB + sentence-transformers
├── vision.py                 # Image quality + local model + Gemini Vision
├── llm.py                    # Gemini wrapper (google-genai)
├── feedback.py               # SQLite interaction + feedback logging
├── knowledge_base.json       # 40+ enriched KB entries
├── requirements.txt          # Python dependencies
├── secrets.toml.example      # Template for Streamlit secrets
├── .gitignore
└── README.md                 # This file
```

---

## 🚀 Quick start (local)

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First install takes a while — `torch`, `transformers`, `sentence-transformers`,
and `chromadb` are all substantial.

### 2. Get a free Gemini API key

Go to https://aistudio.google.com/app/apikey and create a key. Free tier gives
you 15 requests/minute on Gemini 2.0 Flash — plenty for a prototype.

### 3. Set the key

**Local development** — create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

Or export it as an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"   # macOS/Linux
set GEMINI_API_KEY=your-key-here        # Windows
```

### 4. Run it

```bash
streamlit run app.py
```

First launch will download the sentence-transformers model (~90 MB) and, on
first photo upload, the PlantVillage model (~15 MB). Subsequent starts are fast.

---

## ☁️ Deploy to Streamlit Cloud

1. Push the repo to GitHub (make sure `.streamlit/secrets.toml` is NOT committed).
2. Go to https://share.streamlit.io and create a new app from your repo.
3. Main file path: `app.py`
4. After it starts deploying, open **Settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. The app will auto-reload.

### First deploy expectations

- **Cold start: 1–3 minutes.** Streamlit Cloud has to download `torch`,
  `transformers`, `sentence-transformers`, and `chromadb`, then build the
  embedding index. After the first start, subsequent reloads are fast.
- **Memory: tight.** The free Streamlit Cloud tier has ~1 GB RAM. This project
  fits but leaves little headroom. If you hit out-of-memory errors, the first
  thing to drop is the local PlantVillage model in `vision.py` — the app will
  still work with Gemini Vision only.

### Verifying the agent is actually running

After deploy, ask a question and expand the **"Agent trace (debug)"** section
under the answer. You should see:

```
Path: classify → retrieve → check_confidence → synthesize
Classified crop: maize
Top retrieval score: 0.72
```

If you see `Path: classify → retrieve → check_confidence → refuse`, the
retrieval confidence was too low — try a more specific question. If you see
the path but no classified crop, `GEMINI_API_KEY` may not be loading from
secrets.

---

## 🧠 How the agent works

### Classify node
Uses Gemini to extract: which crop, which category, is the question clear
enough to answer? If vague (no crop, no symptoms), returns a clarification
question rather than embedding a hopeless query.

### Retrieve node
Embeds the question (plus any image symptoms) using
`sentence-transformers/all-MiniLM-L6-v2`, searches ChromaDB, filters by crop
metadata. Returns top 4 matches with similarity scores.

### Confidence gate
Three routes:
- **≥ 0.55** — high confidence, go to `synthesize`
- **0.35 – 0.55** — answer but flag for escalation
- **< 0.35** — refuse honestly and escalate to local extension

### Synthesize node
Gemini generates a structured 5-section response (Likely issue, Why, What to
check next, Suggested action, When to seek local support) using **only** the
retrieved sources. Hard rules in the system prompt: never invent information,
never give pesticide doses, always match the farmer's crop.

If the LLM call fails or returns malformed JSON, falls back to the top match
verbatim — the system never hangs.

### Image analysis (separate pipeline)
1. OpenCV Laplacian variance check for blur, mean brightness for exposure.
   Reject bad images with a specific farmer-friendly message ("too blurry —
   please retake").
2. For **maize**: run the local PlantVillage MobileNetV2 model. If confidence
   ≥ 0.60, use its prediction.
3. Otherwise (soybean, or uncertain maize): call Gemini Vision with a
   **describe, don't diagnose** prompt. It returns a list of visible symptoms.
4. The symptoms are injected into the agent's text query, so diagnosis always
   happens through retrieval — never vision alone. This keeps vision and
   knowledge grounded separately and auditable.

---

## 📊 Feedback loop

Every interaction is logged to `agiritechat_feedback.db` (SQLite) with:
- The question, crop hint, classified crop
- Retrieved matches and scores
- Final response
- Full agent trace
- Thumbs up/down when the farmer clicks

Use this to build your evaluation set. A farmer's 👎 is the most valuable
signal you have for improving the knowledge base.

> **Note on Streamlit Cloud:** the SQLite file is on ephemeral storage and
> will be wiped on redeploy. For production, point `feedback.py` at a
> persistent database (Postgres on Neon, Supabase, etc.) or dump the SQLite
> file periodically.

---

## 🗂 Knowledge base schema

Each entry in `knowledge_base.json`:

```json
{
  "id": "maize_yellow_leaves",
  "crop": "maize",
  "category": "nutrient_deficiency",
  "symptoms": ["yellow leaves", "chlorosis", "lower leaves yellow"],
  "growth_stage": ["V3", "V6", "V8"],
  "question": "Why are my maize leaves turning yellow?",
  "answer": "Yellow maize leaves may be caused by...",
  "confidence": "high",
  "source": "general_agronomy"
}
```

Fields:
- `crop`: `"maize"`, `"soybean"`, or `"both"`
- `category`: `pest`, `disease`, `nutrient_deficiency`, `soil`, `weeds`, `drought`, `fertilizer`, `agronomy`, `harvest`, `nodulation`, `water`, `general`
- `symptoms`: helps the retriever match on symptom descriptions
- `growth_stage`: V/R stages where this applies (optional filter)
- `confidence`: your own editorial confidence in the entry
- `source`: for traceability — where the knowledge came from

To add entries: append to the JSON file and push. The ChromaDB index rebuilds
on startup, so no migration step needed.

---

## 🐛 Troubleshooting

**"GEMINI_API_KEY not found"**
Set it in `.streamlit/secrets.toml` locally, or in Streamlit Cloud → Settings → Secrets.

**"Low confidence" on every question**
The knowledge base is small (40 entries). Questions that don't closely match
something in the KB will score low. Expand the KB, or relax `LOW_CONFIDENCE`
in `agent.py` if you want the system to attempt more answers.

**Photo Review says "too blurry" on a photo that looks fine**
Adjust `BLUR_THRESHOLD` in `vision.py`. 80 is conservative for phone photos —
if you want to loosen it, try 40.

**First deploy hangs for several minutes**
Expected. `torch` + `transformers` install is slow on cold start. Wait it out.

**The app works but "AI reasoning: Offline"**
Gemini key is missing or invalid. Check your secrets. The app falls back to
rule-based classification and top-match retrieval, which still works but is
noticeably worse.

---

## 📈 Next steps

- **Expand the KB** to 200+ entries with regional varieties and local pest pressure
- **Add conversation memory** (beyond the current session) with a farm profile
- **Multi-language support** — Gemini handles Swahili, Hausa, Yoruba, Amharic reasonably
- **Voice input/output** via `gTTS` and `whisper` for low-literacy users
- **Move feedback DB to Postgres** for persistent logging across redeploys
- **Build a proper eval set** from thumbs-down interactions and measure before/after changes

---

## 📜 License

Open source for educational and commercial use.

---

**Built for farmers. By developers. For the world.** 🌾
