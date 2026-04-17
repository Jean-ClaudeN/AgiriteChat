"""
app.py — AgiriteChat v3.1 (Warm Editorial Aesthetic)

Changes from v3.0:
- Hero pills replaced with functional category filter buttons (Pests / Diseases / Soil & nutrients)
- Filter state persists across tabs and filters both presets and the knowledge library
- Agent trace hidden by default; shown only when "Developer view" toggle is on in sidebar
- Session ID hidden when developer view is off

Design direction: agricultural editorial. Warm cream + forest green + terracotta.
Fraunces display + DM Sans body.
"""

import uuid
from html import escape

import streamlit as st

from agent import run as run_agent, get_retriever, LANGUAGES
from vision import analyze_field_image
from llm import is_available as llm_available
from feedback import log_interaction, record_feedback, recent_stats, init_db

# ---------------- Page config ----------------
st.set_page_config(
    page_title="AgiriteChat — Crop Advisory",
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded",
)

# ---------------- One-time warm-up ----------------
@st.cache_resource
def _warm_up():
    init_db()
    get_retriever()
    return True

_warm_up()

# ---------------- Session state ----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "language" not in st.session_state:
    st.session_state.language = "en"
if "farmer_profile" not in st.session_state:
    st.session_state.farmer_profile = {
        "name": "", "region": "", "farm_size": "", "crops": "", "planting_date": "",
    }
if "profile_saved" not in st.session_state:
    st.session_state.profile_saved = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "category_filter" not in st.session_state:
    st.session_state.category_filter = None  # None | "pest" | "disease" | "soil"
if "developer_view" not in st.session_state:
    st.session_state.developer_view = False

# ---------------- i18n strings ----------------
UI = {
    "en": {
        "brand_tag": "Crop advisory for smallholder farmers",
        "hero_title": "Practical crop advice,\nrooted in your field.",
        "hero_sub": "AgiriteChat helps maize and soybean farmers identify problems, understand causes, and take action — grounded in agronomic knowledge, personalised to your farm.",
        "filter_pests": "Pests",
        "filter_diseases": "Diseases",
        "filter_soil": "Soil & nutrients",
        "filter_active": "Filtering by",
        "filter_clear": "Clear filter",
        "sidebar_profile": "Your farm profile",
        "sidebar_language": "Language",
        "sidebar_status": "System status",
        "sidebar_dev": "Developer view",
        "sidebar_dev_help": "Show agent reasoning trace (for demos)",
        "name": "Your name",
        "region": "Region / District",
        "farm_size": "Farm size",
        "crops": "Crops you grow",
        "planting_date": "Planting date",
        "save_profile": "Save profile",
        "profile_saved": "Profile saved for this session",
        "profile_note": "Stored only for this session. No account required.",
        "tab_ask": "Ask a question",
        "tab_photo": "Photo review",
        "tab_browse": "Knowledge library",
        "quick_starts": "Quick starts",
        "welcome_default": "Welcome. Ask a question about your maize or soybean field.",
        "welcome_maize": "Welcome, maize farmer. What are you seeing in your field today?",
        "welcome_soybean": "Welcome, soybean farmer. What are you seeing in your field today?",
        "welcome_named": "Welcome, {name}. What are you seeing in your field today?",
        "input_placeholder": "Describe what you see in your field…",
        "analyze_photo": "Analyze photo",
        "photo_upload": "Upload a close-up photo of the affected leaf or plant",
        "photo_desc": "Add a short description (optional)",
        "photo_desc_ph": "e.g. brown lesions on lower leaves, plant is at knee height",
        "symptoms_detected": "What the AI sees in the photo",
        "sources_used": "Sources used",
        "agent_trace": "How the agent answered",
        "confidence_high": "High confidence",
        "confidence_medium": "Medium confidence",
        "confidence_low": "Low confidence",
        "escalation": "This answer has limited confidence. Please confirm with your local extension officer before acting.",
        "helpful": "Helpful",
        "not_helpful": "Not helpful",
        "thanks_feedback": "Thanks for the feedback!",
        "feedback_stats": "Session stats",
        "responsible_use": "Responsible use",
        "responsible_text": "AgiriteChat is a support tool for early interpretation only. Serious disease, pest, or fertility problems should always be confirmed with a local agronomist or extension officer.",
        "library_search": "Search knowledge library",
        "library_search_ph": "yellow lower leaves on maize…",
        # Landing page sections
        "how_kicker": "How it works",
        "how_title": "Three steps from worry to action.",
        "how_step1_title": "Ask",
        "how_step1_desc": "Describe what you see in your field, or upload a photo of the affected plant.",
        "how_step2_title": "Diagnose",
        "how_step2_desc": "AgiriteChat analyzes your input against an agronomic knowledge base.",
        "how_step3_title": "Act",
        "how_step3_desc": "Get clear, practical next steps in your language — and know when to call extension.",
        "impact_kicker": "Real impact",
        "impact_title": "Built for your field, measured by your feedback.",
        "impact_questions": "Questions answered",
        "impact_confident": "Confident answers",
        "impact_languages": "Languages supported",
        "featured_kicker": "Featured topics",
        "featured_title": "What farmers are asking right now.",
        "featured_card_open": "Open this question",
        "languages_kicker": "For every farmer",
        "languages_title": "Available in your language.",
        "languages_sub": "AgiriteChat speaks the languages farmers actually use across East and Central Africa.",
        # Action cards
        "action_ask_title": "Ask a question",
        "action_ask_desc": "Describe what you see in your field and get expert guidance.",
        "action_photo_title": "Upload a photo",
        "action_photo_desc": "Take a photo of the problem and get a diagnosis.",
        # Source labels
        "source_kb": "From knowledge base",
        "source_ai": "Additional AI guidance",
        "source_tag": "Verified source",
    },
    "sw": {
        "brand_tag": "Ushauri wa kilimo kwa wakulima wadogo",
        "hero_title": "Ushauri wa vitendo wa mazao,\nuliojikita katika shamba lako.",
        "hero_sub": "AgiriteChat inasaidia wakulima wa mahindi na soya kutambua matatizo, kuelewa sababu, na kuchukua hatua — ikizingatia maarifa ya kilimo, iliyobinafsishwa kwa shamba lako.",
        "filter_pests": "Wadudu",
        "filter_diseases": "Magonjwa",
        "filter_soil": "Udongo na virutubisho",
        "filter_active": "Inachuja kwa",
        "filter_clear": "Ondoa kichujio",
        "sidebar_profile": "Wasifu wa shamba lako",
        "sidebar_language": "Lugha",
        "sidebar_status": "Hali ya mfumo",
        "sidebar_dev": "Hali ya msanidi",
        "sidebar_dev_help": "Onyesha maelezo ya wakala (kwa maonyesho)",
        "name": "Jina lako",
        "region": "Mkoa / Wilaya",
        "farm_size": "Ukubwa wa shamba",
        "crops": "Mazao unayolima",
        "planting_date": "Tarehe ya kupanda",
        "save_profile": "Hifadhi wasifu",
        "profile_saved": "Wasifu umehifadhiwa kwa kipindi hiki",
        "profile_note": "Imehifadhiwa kwa kipindi hiki tu. Hakuna akaunti inayohitajika.",
        "tab_ask": "Uliza swali",
        "tab_photo": "Kagua picha",
        "tab_browse": "Maktaba ya maarifa",
        "quick_starts": "Maswali ya haraka",
        "welcome_default": "Karibu. Uliza swali kuhusu shamba lako la mahindi au soya.",
        "welcome_maize": "Karibu, mkulima wa mahindi. Unaona nini shambani mwako leo?",
        "welcome_soybean": "Karibu, mkulima wa soya. Unaona nini shambani mwako leo?",
        "welcome_named": "Karibu, {name}. Unaona nini shambani mwako leo?",
        "input_placeholder": "Eleza unachoona shambani mwako…",
        "analyze_photo": "Chambua picha",
        "photo_upload": "Pakia picha ya karibu ya jani au mmea ulioathirika",
        "photo_desc": "Ongeza maelezo mafupi (hiari)",
        "photo_desc_ph": "mfano: madoa ya kahawia kwenye majani ya chini",
        "symptoms_detected": "Kile AI inachokiona kwenye picha",
        "sources_used": "Vyanzo vilivyotumika",
        "agent_trace": "Jinsi wakala alivyojibu",
        "confidence_high": "Uhakika wa juu",
        "confidence_medium": "Uhakika wa wastani",
        "confidence_low": "Uhakika wa chini",
        "escalation": "Jibu hili lina uhakika mdogo. Tafadhali thibitisha na afisa wa ugani kabla ya kuchukua hatua.",
        "helpful": "Lilikuwa la msaada",
        "not_helpful": "Halikuwa la msaada",
        "thanks_feedback": "Asante kwa maoni!",
        "feedback_stats": "Takwimu za kipindi",
        "responsible_use": "Matumizi yenye uwajibikaji",
        "responsible_text": "AgiriteChat ni chombo cha msaada cha tafsiri ya mapema tu. Matatizo makubwa ya magonjwa, wadudu, au rutuba yanapaswa kuthibitishwa na mtaalam wa kilimo wa ndani.",
        "library_search": "Tafuta katika maktaba",
        "library_search_ph": "majani ya njano ya chini ya mahindi…",
        "how_kicker": "Jinsi inavyofanya kazi",
        "how_title": "Hatua tatu kutoka wasiwasi hadi hatua.",
        "how_step1_title": "Uliza",
        "how_step1_desc": "Eleza unachoona shambani, au pakia picha ya mmea ulioathirika.",
        "how_step2_title": "Tambua",
        "how_step2_desc": "AgiriteChat inachambua mchango wako dhidi ya hifadhi ya maarifa ya kilimo.",
        "how_step3_title": "Tenda",
        "how_step3_desc": "Pata hatua wazi za vitendo katika lugha yako — na ujue wakati wa kupiga simu kwa ugani.",
        "impact_kicker": "Athari halisi",
        "impact_title": "Imejengwa kwa shamba lako, imepimwa kwa maoni yako.",
        "impact_questions": "Maswali yaliyojibiwa",
        "impact_confident": "Majibu ya uhakika",
        "impact_languages": "Lugha zinazoungwa mkono",
        "featured_kicker": "Mada zinazoangaziwa",
        "featured_title": "Kile wakulima wanachouliza sasa hivi.",
        "featured_card_open": "Fungua swali hili",
        "languages_kicker": "Kwa kila mkulima",
        "languages_title": "Inapatikana katika lugha yako.",
        "languages_sub": "AgiriteChat inazungumza lugha ambazo wakulima wanazitumia Afrika Mashariki na Kati.",
        "action_ask_title": "Uliza swali",
        "action_ask_desc": "Eleza unachoona shambani na upate ushauri wa kitaalamu.",
        "action_photo_title": "Pakia picha",
        "action_photo_desc": "Piga picha ya tatizo na upate uchunguzi.",
        "source_kb": "Kutoka kwa msingi wa maarifa",
        "source_ai": "Mwongozo wa ziada wa AI",
        "source_tag": "Chanzo kilichothibitishwa",
    },
    "fr": {
        "brand_tag": "Conseil agricole pour petits exploitants",
        "hero_title": "Des conseils pratiques,\nenracinés dans votre champ.",
        "hero_sub": "AgiriteChat aide les cultivateurs de maïs et de soja à identifier les problèmes, comprendre les causes et agir — avec un savoir agronomique ancré, personnalisé pour votre exploitation.",
        "filter_pests": "Ravageurs",
        "filter_diseases": "Maladies",
        "filter_soil": "Sol et nutriments",
        "filter_active": "Filtré par",
        "filter_clear": "Effacer le filtre",
        "sidebar_profile": "Votre profil agricole",
        "sidebar_language": "Langue",
        "sidebar_status": "État du système",
        "sidebar_dev": "Vue développeur",
        "sidebar_dev_help": "Afficher le raisonnement de l'agent (pour les démos)",
        "name": "Votre nom",
        "region": "Région / District",
        "farm_size": "Taille de l'exploitation",
        "crops": "Cultures pratiquées",
        "planting_date": "Date de semis",
        "save_profile": "Enregistrer le profil",
        "profile_saved": "Profil enregistré pour cette session",
        "profile_note": "Conservé uniquement pour cette session. Pas de compte requis.",
        "tab_ask": "Poser une question",
        "tab_photo": "Analyse de photo",
        "tab_browse": "Bibliothèque",
        "quick_starts": "Démarrage rapide",
        "welcome_default": "Bienvenue. Posez une question sur votre champ de maïs ou de soja.",
        "welcome_maize": "Bienvenue, cultivateur de maïs. Que voyez-vous dans votre champ aujourd'hui ?",
        "welcome_soybean": "Bienvenue, cultivateur de soja. Que voyez-vous dans votre champ aujourd'hui ?",
        "welcome_named": "Bienvenue, {name}. Que voyez-vous dans votre champ aujourd'hui ?",
        "input_placeholder": "Décrivez ce que vous voyez dans votre champ…",
        "analyze_photo": "Analyser la photo",
        "photo_upload": "Téléchargez une photo en gros plan de la feuille ou de la plante affectée",
        "photo_desc": "Ajoutez une courte description (optionnel)",
        "photo_desc_ph": "ex. lésions brunes sur les feuilles basses",
        "symptoms_detected": "Ce que l'IA voit sur la photo",
        "sources_used": "Sources utilisées",
        "agent_trace": "Comment l'agent a répondu",
        "confidence_high": "Confiance élevée",
        "confidence_medium": "Confiance moyenne",
        "confidence_low": "Confiance faible",
        "escalation": "Cette réponse a une confiance limitée. Veuillez confirmer avec votre agent de vulgarisation avant d'agir.",
        "helpful": "Utile",
        "not_helpful": "Peu utile",
        "thanks_feedback": "Merci pour votre retour !",
        "feedback_stats": "Statistiques de session",
        "responsible_use": "Utilisation responsable",
        "responsible_text": "AgiriteChat est un outil d'aide à l'interprétation précoce. Les problèmes graves de maladies, ravageurs ou fertilité doivent toujours être confirmés par un agronome local.",
        "library_search": "Rechercher dans la bibliothèque",
        "library_search_ph": "feuilles basses jaunes de maïs…",
        "how_kicker": "Comment ça marche",
        "how_title": "Trois étapes du souci à l'action.",
        "how_step1_title": "Demander",
        "how_step1_desc": "Décrivez ce que vous voyez dans votre champ, ou téléchargez une photo de la plante affectée.",
        "how_step2_title": "Diagnostiquer",
        "how_step2_desc": "AgiriteChat analyse votre demande dans une base de connaissances agronomiques.",
        "how_step3_title": "Agir",
        "how_step3_desc": "Recevez des étapes claires et pratiques dans votre langue — et sachez quand contacter la vulgarisation.",
        "impact_kicker": "Impact réel",
        "impact_title": "Conçu pour votre champ, mesuré par vos retours.",
        "impact_questions": "Questions répondues",
        "impact_confident": "Réponses confiantes",
        "impact_languages": "Langues prises en charge",
        "featured_kicker": "Sujets en vedette",
        "featured_title": "Ce que les agriculteurs demandent en ce moment.",
        "featured_card_open": "Ouvrir cette question",
        "languages_kicker": "Pour chaque agriculteur",
        "languages_title": "Disponible dans votre langue.",
        "languages_sub": "AgiriteChat parle les langues que les agriculteurs utilisent réellement en Afrique de l'Est et centrale.",
        "action_ask_title": "Poser une question",
        "action_ask_desc": "Décrivez ce que vous voyez et recevez des conseils d'expert.",
        "action_photo_title": "Envoyer une photo",
        "action_photo_desc": "Prenez une photo du problème et obtenez un diagnostic.",
        "source_kb": "De la base de connaissances",
        "source_ai": "Conseils AI supplémentaires",
        "source_tag": "Source vérifiée",
    },
    "rw": {
        "brand_tag": "Inama z'ubuhinzi ku bahinzi bato",
        "hero_title": "Inama z'ubuhinzi zifatika,\nzishingiye ku murima wawe.",
        "hero_sub": "AgiriteChat ifasha abahinzi b'ibigori n'ibishyimbo kumenya ibibazo, gusobanukirwa impamvu, no gufata ibyemezo — ishingiye ku bumenyi bw'ubuhinzi, ihariwe umurima wawe.",
        "filter_pests": "Udukoko",
        "filter_diseases": "Indwara",
        "filter_soil": "Ubutaka n'intungamubiri",
        "filter_active": "Biyungurura ku",
        "filter_clear": "Siba iyungurura",
        "sidebar_profile": "Umwirondoro w'umurima wawe",
        "sidebar_language": "Ururimi",
        "sidebar_status": "Imimerere ya sisitemu",
        "sidebar_dev": "Uburyo bw'umukoraporogaramu",
        "sidebar_dev_help": "Erekana uburyo agent ikora (ku magaragaza)",
        "name": "Izina ryawe",
        "region": "Intara / Akarere",
        "farm_size": "Ingano y'umurima",
        "crops": "Ibihingwa uhinga",
        "planting_date": "Itariki yo gutera",
        "save_profile": "Bika umwirondoro",
        "profile_saved": "Umwirondoro wabitswe muri iki gihe",
        "profile_note": "Bibitswe muri iki gihe gusa. Nta konti ikenewe.",
        "tab_ask": "Baza ikibazo",
        "tab_photo": "Isesengura ry'ifoto",
        "tab_browse": "Ububiko bw'ubumenyi",
        "quick_starts": "Gutangira vuba",
        "welcome_default": "Murakaza neza. Baza ikibazo ku murima wawe w'ibigori cyangwa ibishyimbo.",
        "welcome_maize": "Murakaza neza, umuhinzi w'ibigori. Uravye iki mu murima wawe uyu munsi?",
        "welcome_soybean": "Murakaza neza, umuhinzi w'ibishyimbo. Uravye iki mu murima wawe uyu munsi?",
        "welcome_named": "Murakaza neza, {name}. Uravye iki mu murima wawe uyu munsi?",
        "input_placeholder": "Sobanura ibyo urabona mu murima wawe…",
        "analyze_photo": "Sesengura ifoto",
        "photo_upload": "Shyiraho ifoto y'ibabi cyangwa igihingwa cyanduye",
        "photo_desc": "Ongeraho ibisobanuro bigufi (bishoboka)",
        "photo_desc_ph": "urugero: utuntu two mu ibara ry'umukara",
        "symptoms_detected": "Ibyo AI ireba kuri iyi foto",
        "sources_used": "Inkomoko zakoreshejwe",
        "agent_trace": "Uburyo agent yasubije",
        "confidence_high": "Ikizere gikomeye",
        "confidence_medium": "Ikizere rwagati",
        "confidence_low": "Ikizere gike",
        "escalation": "Iyi nyishyu ifite ikizere gike. Nyamuneka bimenyeshe umukozi w'ubuhinzi mbere yo gufata icyemezo.",
        "helpful": "Byanyunguye",
        "not_helpful": "Ntibyanyunguye",
        "thanks_feedback": "Murakoze ku gitekerezo!",
        "feedback_stats": "Imibare y'iki gihe",
        "responsible_use": "Imikoreshereze y'inshingano",
        "responsible_text": "AgiriteChat ni igikoresho cy'ubufasha gusa. Ibibazo bikomeye by'indwara, udukoko, cyangwa umwanda bigomba kwemezwa n'umuhanga mu buhinzi wa hafi.",
        "library_search": "Shakisha mu bubiko",
        "library_search_ph": "amababi y'umuhondo y'ibigori…",
        "how_kicker": "Uko bikora",
        "how_title": "Intambwe eshatu uva ku guhangayika ujya ku gukora.",
        "how_step1_title": "Baza",
        "how_step1_desc": "Sobanura ibyo urabona mu murima wawe, cyangwa shyiraho ifoto y'igihingwa cyanduye.",
        "how_step2_title": "Suzuma",
        "how_step2_desc": "AgiriteChat isuzuma ibyo wavuze ihereye ku bumenyi bw'ubuhinzi.",
        "how_step3_title": "Korera",
        "how_step3_desc": "Bonera intambwe zigaragara mu rurimi rwawe — kandi umenye igihe cyo guhamagara umufasha.",
        "impact_kicker": "Ingaruka nyazo",
        "impact_title": "Yakozwe ku murima wawe, ipimwe n'ibitekerezo byawe.",
        "impact_questions": "Ibibazo byashubijwe",
        "impact_confident": "Ibisubizo by'ikizere",
        "impact_languages": "Indimi zifashishwa",
        "featured_kicker": "Ingingo zihariwe",
        "featured_title": "Ibyo abahinzi babaza ubu.",
        "featured_card_open": "Fungura iki kibazo",
        "languages_kicker": "Ku muhinzi wese",
        "languages_title": "Iraboneka mu rurimi rwawe.",
        "languages_sub": "AgiriteChat ivuga indimi abahinzi bakoresha mu Afurika y'Iburasirazuba no Hagati.",
        "action_ask_title": "Baza ikibazo",
        "action_ask_desc": "Sobanura ibyo ubona mu murima wawe ubonere inama z'inzobere.",
        "action_photo_title": "Shyiraho ifoto",
        "action_photo_desc": "Fata ifoto y'ikibazo ubonere isuzuma.",
        "source_kb": "Biturutse ku bumenyi bwujujwe",
        "source_ai": "Inama z'inyongera za AI",
        "source_tag": "Isoko ryemejwe",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return UI.get(lang, UI["en"]).get(key, UI["en"].get(key, key))


# ---------------- Category mapping ----------------
# Maps filter button names to KB category tags
CATEGORY_MAP = {
    "pest": ["pest"],
    "disease": ["disease"],
    "soil": ["soil", "nutrient_deficiency", "fertilizer"],
}


# ---------------- Crop-specific preset questions ----------------
# Each preset has: (label, question, category) — category enables filtering
PRESETS = {
    "maize": {
        "en": [
            ("🟡 Yellow lower leaves", "Why are the lower leaves on my maize turning yellow?", "soil"),
            ("🟣 Purple leaves", "My maize leaves are turning purple. What is wrong?", "soil"),
            ("🐛 Fall armyworm", "How do I control fall armyworm in maize?", "pest"),
            ("🍃 Leaf blight", "What are the signs of maize leaf blight?", "disease"),
        ],
        "sw": [
            ("🟡 Majani ya chini ya njano", "Kwa nini majani ya chini ya mahindi yanageuka njano?", "soil"),
            ("🟣 Majani ya zambarau", "Majani ya mahindi yangu yanageuka zambarau. Kuna shida gani?", "soil"),
            ("🐛 Viwavi wa majani", "Nidhibiti vipi viwavi wa majani katika mahindi?", "pest"),
            ("🍃 Ugonjwa wa majani", "Dalili za ugonjwa wa majani ya mahindi ni zipi?", "disease"),
        ],
        "fr": [
            ("🟡 Feuilles basses jaunes", "Pourquoi les feuilles basses de mon maïs jaunissent-elles ?", "soil"),
            ("🟣 Feuilles violettes", "Les feuilles de mon maïs deviennent violettes. Qu'est-ce qui ne va pas ?", "soil"),
            ("🐛 Chenille légionnaire", "Comment lutter contre la chenille légionnaire d'automne sur le maïs ?", "pest"),
            ("🍃 Brûlure des feuilles", "Quels sont les signes de la brûlure des feuilles du maïs ?", "disease"),
        ],
        "rw": [
            ("🟡 Amababi yo hasi y'umuhondo", "Kubera iki amababi yo hasi y'ibigori byanjye ahinduka umuhondo?", "soil"),
            ("🟣 Amababi y'umutuku", "Amababi y'ibigori byanjye ahinduka umutuku. Iki kibazo ni iki?", "soil"),
            ("🐛 Udukoko tw'amababi", "Nigute nakurikirana udukoko tw'amababi mu bigori?", "pest"),
            ("🍃 Indwara y'amababi", "Ni ibihe bimenyetso by'indwara y'amababi y'ibigori?", "disease"),
        ],
    },
    "soybean": {
        "en": [
            ("💧 Not fixing nitrogen", "My soybeans are weak and not fixing nitrogen well. What could be wrong?", "soil"),
            ("🌱 Poor nodulation", "How can I improve soybean nodulation?", "soil"),
            ("🦠 Root rot", "What causes root rot in soybean?", "disease"),
            ("🟡 Yellow leaves", "Why are my soybean leaves turning yellow?", "soil"),
        ],
        "sw": [
            ("💧 Hazifungi naitrojeni", "Soya zangu ni dhaifu na hazifungi naitrojeni vizuri. Kuna shida gani?", "soil"),
            ("🌱 Unodushaji mbaya", "Ninawezaje kuboresha unodushaji wa soya?", "soil"),
            ("🦠 Uozo wa mizizi", "Ni nini husababisha uozo wa mizizi katika soya?", "disease"),
            ("🟡 Majani ya njano", "Kwa nini majani ya soya yangu yanageuka njano?", "soil"),
        ],
        "fr": [
            ("💧 Pas de fixation d'azote", "Mes sojas sont faibles et ne fixent pas bien l'azote. Qu'est-ce qui ne va pas ?", "soil"),
            ("🌱 Mauvaise nodulation", "Comment améliorer la nodulation du soja ?", "soil"),
            ("🦠 Pourriture racinaire", "Qu'est-ce qui cause la pourriture racinaire chez le soja ?", "disease"),
            ("🟡 Feuilles jaunes", "Pourquoi mes feuilles de soja deviennent-elles jaunes ?", "soil"),
        ],
        "rw": [
            ("💧 Ntizikora azote", "Ibishyimbo byanjye birananiwe ntibikora azote neza. Ni ikihe kibazo?", "soil"),
            ("🌱 Noduli nke", "Nigute nahindura noduli z'ibishyimbo?", "soil"),
            ("🦠 Kubora kw'imizi", "Ni iki gitera kubora kw'imizi y'ibishyimbo?", "disease"),
            ("🟡 Amababi y'umuhondo", "Kubera iki amababi y'ibishyimbo byanjye ahinduka umuhondo?", "soil"),
        ],
    },
    "general": {
        "en": [
            ("🌽 Maize leaf blight", "How do I manage maize leaf blight?", "disease"),
            ("🟣 Purple maize leaves", "My maize leaves are turning purple. What could be wrong?", "soil"),
            ("🫘 Soybean nitrogen", "My soybeans are weak and not fixing nitrogen. What could be wrong?", "soil"),
            ("🦠 Soybean root rot", "What causes root rot in soybean?", "disease"),
            ("🐛 Fall armyworm", "How do I control fall armyworm in maize?", "pest"),
            ("🌾 Soybean pod borer", "How do I manage soybean pod borers?", "pest"),
        ],
        "sw": [
            ("🌽 Ugonjwa wa majani", "Ninawezaje kudhibiti ugonjwa wa majani ya mahindi?", "disease"),
            ("🟣 Majani ya zambarau", "Majani ya mahindi yangu yanageuka zambarau. Kuna shida gani?", "soil"),
            ("🫘 Naitrojeni ya soya", "Soya zangu ni dhaifu na hazifungi naitrojeni. Kuna shida gani?", "soil"),
            ("🦠 Uozo wa mizizi", "Ni nini husababisha uozo wa mizizi katika soya?", "disease"),
            ("🐛 Viwavi wa majani", "Nidhibiti vipi viwavi wa majani katika mahindi?", "pest"),
            ("🌾 Wadudu wa maganda", "Nidhibiti vipi wadudu wa maganda ya soya?", "pest"),
        ],
        "fr": [
            ("🌽 Brûlure du maïs", "Comment gérer la brûlure des feuilles du maïs ?", "disease"),
            ("🟣 Maïs violet", "Mes feuilles de maïs deviennent violettes. Qu'est-ce qui ne va pas ?", "soil"),
            ("🫘 Azote du soja", "Mes sojas sont faibles et ne fixent pas l'azote. Qu'est-ce qui ne va pas ?", "soil"),
            ("🦠 Pourriture du soja", "Qu'est-ce qui cause la pourriture racinaire du soja ?", "disease"),
            ("🐛 Chenille légionnaire", "Comment lutter contre la chenille légionnaire sur le maïs ?", "pest"),
            ("🌾 Borer du soja", "Comment gérer les borers de gousses de soja ?", "pest"),
        ],
        "rw": [
            ("🌽 Indwara y'ibigori", "Nigute nakurikirana indwara y'amababi y'ibigori?", "disease"),
            ("🟣 Amababi y'umutuku", "Amababi y'ibigori byanjye ahinduka umutuku. Ni iki kibazo?", "soil"),
            ("🫘 Azote y'ibishyimbo", "Ibishyimbo byanjye birananiwe ntibikora azote. Ni iki kibazo?", "soil"),
            ("🦠 Kubora kw'imizi", "Ni iki gitera kubora kw'imizi y'ibishyimbo?", "disease"),
            ("🐛 Udukoko tw'amababi", "Nigute nakurikirana udukoko tw'amababi mu bigori?", "pest"),
            ("🌾 Udukoko tw'ibiryo", "Nigute nakurikirana udukoko tw'ibiryo by'ibishyimbo?", "pest"),
        ],
    },
}


# ---------------- CSS ----------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --cream: #faf6ee;
    --cream-dark: #f2ecdd;
    --ink: #1a1d14;
    --ink-soft: #5a5d50;
    --ink-mute: #8a8d7e;
    --forest-900: #1f3a26;
    --forest-700: #2d5234;
    --forest-500: #4a7c52;
    --forest-100: #e6efe4;
    --terracotta: #c65a3a;
    --terracotta-dark: #9a3f23;
    --terracotta-light: #f4dfd4;
    --gold: #c8984a;
    --border: #e4ddc9;
    --shadow-warm: 0 2px 12px rgba(90, 70, 30, 0.08), 0 1px 3px rgba(90, 70, 30, 0.05);
    --shadow-lifted: 0 12px 40px rgba(90, 70, 30, 0.12), 0 2px 8px rgba(90, 70, 30, 0.06);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
    color: var(--ink) !important;
    background: var(--cream) !important;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1180px !important;
}

section[data-testid="stSidebar"] {
    background: var(--cream-dark) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stDateInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--forest-700) !important;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.brand-group { display: flex; align-items: center; gap: 0.8rem; }
.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: var(--forest-700);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--cream);
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 700;
}
.brand-name {
    font-family: 'Fraunces', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--forest-900);
    letter-spacing: -0.02em;
    line-height: 1;
}
.brand-tag {
    font-size: 0.78rem;
    color: var(--ink-soft);
    font-weight: 500;
    letter-spacing: 0.02em;
    margin-top: 2px;
}
.topbar-session {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* Hero */
.hero {
    background:
        linear-gradient(135deg, rgba(31, 58, 38, 0.88) 0%, rgba(45, 82, 52, 0.78) 55%, rgba(61, 101, 64, 0.7) 100%),
        url('https://images.pexels.com/photos/13525130/pexels-photo-13525130.jpeg?auto=compress&cs=tinysrgb&w=1600') center/cover no-repeat,
        linear-gradient(135deg, var(--forest-900) 0%, var(--forest-700) 55%, #3d6540 100%);
    background-blend-mode: normal;
    border-radius: 20px;
    padding: 2.5rem 2.8rem 2.2rem 2.8rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lifted);
}
.hero::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(200, 152, 74, 0.25) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(198, 90, 58, 0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero-label {
    display: inline-block;
    background: rgba(250, 246, 238, 0.15);
    color: var(--cream);
    border: 1px solid rgba(250, 246, 238, 0.25);
    padding: 0.4rem 0.9rem;
    border-radius: 100px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    position: relative;
    z-index: 1;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 4.5vw, 3.2rem);
    font-weight: 700;
    line-height: 1.05;
    color: var(--cream);
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    white-space: pre-line;
    position: relative;
    z-index: 1;
    max-width: 680px;
}
.hero-sub {
    font-size: 1.02rem;
    line-height: 1.6;
    color: rgba(250, 246, 238, 0.88);
    max-width: 620px;
    position: relative;
    z-index: 1;
}

/* Category filter bar — below hero, above tabs */
.category-bar {
    display: flex;
    gap: 0.7rem;
    align-items: center;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.category-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: 0.3rem;
}
.active-filter-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--terracotta-light);
    border: 1px solid var(--terracotta);
    color: var(--terracotta-dark);
    padding: 0.35rem 0.75rem;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* Welcome card */
.welcome-card {
    background: var(--cream);
    border: 1px solid var(--border);
    border-left: 4px solid var(--terracotta);
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: var(--shadow-warm);
}
.welcome-avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--terracotta-light);
    color: var(--terracotta-dark);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-size: 1.2rem;
    font-weight: 700;
    flex-shrink: 0;
}
.welcome-text {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 500;
    color: var(--forest-900);
    line-height: 1.3;
}
.welcome-meta {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: var(--ink-soft);
    font-weight: 400;
    margin-top: 2px;
}

.section-header {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--forest-900);
    margin: 0.3rem 0 0.2rem 0;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.92rem;
    color: var(--ink-soft);
    margin-bottom: 1.2rem;
}

/* Streamlit buttons (default styling for presets and filters) */
div.stButton > button {
    background: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.7rem 1rem !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(90, 70, 30, 0.04) !important;
    height: auto !important;
    min-height: 48px !important;
}
div.stButton > button:hover {
    border-color: var(--forest-500) !important;
    background: var(--forest-100) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-warm) !important;
}
div.stButton > button:active, div.stButton > button:focus {
    border-color: var(--forest-700) !important;
}
div.stButton > button[kind="primary"] {
    background: var(--terracotta) !important;
    border-color: var(--terracotta-dark) !important;
    color: var(--cream) !important;
    font-weight: 600 !important;
    text-align: center !important;
}
div.stButton > button[kind="primary"]:hover {
    background: var(--terracotta-dark) !important;
}

/* Answer card */
.answer-card {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-warm);
    position: relative;
}
.answer-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 1.8rem;
    right: 1.8rem;
    height: 3px;
    background: linear-gradient(90deg, var(--forest-700), var(--terracotta), var(--gold));
    border-radius: 0 0 3px 3px;
}
.answer-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
    gap: 0.8rem;
    flex-wrap: wrap;
}
.answer-kicker {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.conf-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
}
.conf-high { background: var(--forest-100); color: var(--forest-900); border: 1px solid #c3dac2; }
.conf-medium { background: #fdf4e1; color: #7a5a00; border: 1px solid #ead9a7; }
.conf-low { background: var(--terracotta-light); color: var(--terracotta-dark); border: 1px solid #eac5b5; }

.answer-issue {
    font-family: 'Fraunces', serif;
    font-size: 1.55rem;
    font-weight: 600;
    line-height: 1.2;
    color: var(--forest-900);
    margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
}
.answer-section { margin-bottom: 1.1rem; }
.answer-section:last-child { margin-bottom: 0; }
.answer-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--terracotta-dark);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}
.answer-body {
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--ink);
}

.escalate-box {
    background: #fff5e8;
    border: 1px solid #ead9a7;
    border-left: 4px solid var(--gold);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #6b4a10;
}

.symptoms-box {
    background: var(--forest-100);
    border: 1px solid #c8dcc6;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.symptoms-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--forest-700);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.footer-card {
    background: var(--cream-dark);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 2rem;
    font-size: 0.88rem;
    color: var(--ink-soft);
    line-height: 1.6;
}
.footer-card strong {
    font-family: 'Fraunces', serif;
    font-size: 1.0rem;
    color: var(--forest-900);
    display: block;
    margin-bottom: 0.35rem;
    font-weight: 600;
}

/* Streamlit tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--ink-soft) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--forest-900) !important;
    border-bottom: 2px solid var(--terracotta) !important;
}

.stChatInput textarea, .stChatInput input {
    background: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.98rem !important;
    padding: 0.8rem 1rem !important;
}

details[data-testid="stExpander"] {
    background: var(--cream) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin: 0.5rem 0 !important;
}
details[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: var(--forest-900) !important;
    padding: 0.8rem 1rem !important;
}

/* ============================================================ */
/*  LANDING PAGE SECTIONS (v3.3)                                  */
/* ============================================================ */

/* Section spacing — used by all landing sections */
.land-section {
    margin: 2.5rem 0 2rem 0;
}
.land-kicker {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--terracotta-dark);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.4rem;
}
.land-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(1.6rem, 3vw, 2.1rem);
    font-weight: 700;
    color: var(--forest-900);
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 1.5rem;
    max-width: 640px;
}
.land-sub {
    font-size: 1rem;
    line-height: 1.6;
    color: var(--ink-soft);
    max-width: 600px;
    margin-top: -0.8rem;
    margin-bottom: 1.5rem;
}

/* "How it works" — three step cards */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
}
.how-card {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.4rem 1.4rem 1.5rem 1.4rem;
    box-shadow: var(--shadow-warm);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.how-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lifted);
}
.how-card-image {
    width: 100%;
    height: 140px;
    border-radius: 12px;
    background-size: cover;
    background-position: center;
    margin-bottom: 1rem;
    position: relative;
}
.how-card-image::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 12px;
    background: linear-gradient(180deg, transparent 50%, rgba(31, 58, 38, 0.35) 100%);
}
.how-step-number {
    font-family: 'Fraunces', serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--terracotta-dark);
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.how-step-title {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--forest-900);
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}
.how-step-desc {
    font-size: 0.93rem;
    line-height: 1.55;
    color: var(--ink-soft);
}

/* "Real impact" — three big numbers */
.impact-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
}
.impact-card {
    background: linear-gradient(160deg, var(--forest-700) 0%, var(--forest-900) 100%);
    border-radius: 18px;
    padding: 1.8rem 1.6rem 1.6rem 1.6rem;
    color: var(--cream);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-warm);
}
.impact-card.terracotta {
    background: linear-gradient(160deg, var(--terracotta) 0%, var(--terracotta-dark) 100%);
}
.impact-card.gold {
    background: linear-gradient(160deg, var(--gold) 0%, #a17a36 100%);
}
.impact-card::before {
    content: "";
    position: absolute;
    top: -50px;
    right: -50px;
    width: 180px;
    height: 180px;
    background: radial-gradient(circle, rgba(250, 246, 238, 0.15) 0%, transparent 65%);
}
.impact-number {
    font-family: 'Fraunces', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.03em;
    position: relative;
    z-index: 1;
}
.impact-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.7rem;
    opacity: 0.92;
    position: relative;
    z-index: 1;
}

/* "Featured topics" — three real KB cards */
.featured-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
}
.featured-card {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    box-shadow: var(--shadow-warm);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
}
.featured-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lifted);
}
.featured-image {
    width: 100%;
    height: 160px;
    background-size: cover;
    background-position: center;
    position: relative;
}
.featured-image::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, transparent 40%, rgba(31, 58, 38, 0.5) 100%);
}
.featured-tag {
    position: absolute;
    top: 0.8rem;
    left: 0.8rem;
    background: rgba(250, 246, 238, 0.92);
    color: var(--forest-900);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.65rem;
    border-radius: 100px;
    z-index: 1;
}
.featured-body {
    padding: 1.2rem 1.3rem 1.4rem 1.3rem;
    flex: 1;
    display: flex;
    flex-direction: column;
}
.featured-title {
    font-family: 'Fraunces', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--forest-900);
    line-height: 1.25;
    margin-bottom: 0.5rem;
}
.featured-summary {
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--ink-soft);
    margin-bottom: 1rem;
    flex: 1;
}
.featured-cta {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--terracotta-dark);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* "Languages" strip */
.lang-section {
    background: linear-gradient(180deg, var(--cream-dark) 0%, var(--cream) 100%);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    margin: 2.5rem 0 2rem 0;
}
.lang-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem;
    margin-top: 1.4rem;
}
.lang-pill {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1rem;
    text-align: center;
    transition: all 0.2s ease;
}
.lang-pill:hover {
    border-color: var(--forest-500);
    transform: translateY(-1px);
}
.lang-pill-name {
    font-family: 'Fraunces', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--forest-900);
    margin-bottom: 0.2rem;
}
.lang-pill-native {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: var(--ink-mute);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.lang-pill.beta::after {
    content: "BETA";
    display: inline-block;
    margin-left: 0.4rem;
    background: var(--gold);
    color: var(--cream);
    font-size: 0.62rem;
    font-weight: 700;
    padding: 0.1rem 0.4rem;
    border-radius: 100px;
    vertical-align: middle;
    letter-spacing: 0.08em;
}

/* SVG icon system — replaces emojis */
.svg-icon {
    display: inline-block;
    width: 18px;
    height: 18px;
    vertical-align: middle;
    margin-right: 0.4rem;
    opacity: 0.9;
}

/* ── BIG ACTION CARDS (Ask / Upload) ── */
.action-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0 1rem 0;
}
.action-card {
    background: var(--forest-900);
    border-radius: 18px;
    padding: 1.8rem 1.6rem;
    color: var(--cream);
    position: relative;
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: var(--shadow-warm);
}
.action-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lifted);
}
.action-card.terra {
    background: linear-gradient(160deg, var(--terracotta) 0%, var(--terracotta-dark) 100%);
}
.action-card::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(250,246,238,0.12) 0%, transparent 65%);
}
.action-icon {
    font-size: 2.2rem;
    margin-bottom: 0.8rem;
    display: block;
}
.action-title {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}
.action-desc {
    font-size: 0.92rem;
    line-height: 1.5;
    opacity: 0.85;
}

/* ── SOURCE SEPARATION in answers ── */
.source-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 100px;
    margin-bottom: 0.6rem;
}
.source-badge.kb {
    background: var(--forest-100);
    color: var(--forest-700);
    border: 1px solid #c8dcc6;
}
.source-badge.ai {
    background: #fff5e8;
    color: #6b4a10;
    border: 1px solid #ead9a7;
}
.ai-context-section {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px dashed var(--border);
}
.conversational-msg {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--ink);
    box-shadow: var(--shadow-warm);
}

/* Inline SVG icons used in filter buttons via background-image */
.icon-bug, .icon-leaf, .icon-sprout, .icon-globe, .icon-wheat, .icon-arrow {
    display: inline-block;
    width: 16px;
    height: 16px;
    background-repeat: no-repeat;
    background-position: center;
    background-size: contain;
    vertical-align: -3px;
    margin-right: 0.45rem;
}
.icon-bug {
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%239a3f23' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M8 2l1.88 1.88'/><path d='M14.12 3.88L16 2'/><path d='M9 7.13v-1a3.003 3.003 0 116 0v1'/><path d='M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 014-4h4a4 4 0 014 4v3c0 3.3-2.7 6-6 6zM12 20v-9M6.53 9C4.6 8.8 3 7.1 3 5M6 13H2M3 21c0-2.1 1.7-3.9 3.8-4M20.97 5c0 2.1-1.6 3.8-3.5 4M22 13h-4M17.2 17c2.1.1 3.8 1.9 3.8 4'/></svg>");
}
.icon-leaf {
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%239a3f23' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19.2 2.96a1 1 0 0 1 1.8.5c0 8.42-3.42 14.66-9.4 16.4Z'/><path d='M2 21c0-3 1.85-5.36 5.08-6'/></svg>");
}
.icon-sprout {
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%239a3f23' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M7 20h10'/><path d='M10 20c5.5-2.5.8-6.4 3-10'/><path d='M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z'/><path d='M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z'/></svg>");
}
.icon-globe {
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%232d5234' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='12' cy='12' r='10'/><line x1='2' y1='12' x2='22' y2='12'/><path d='M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z'/></svg>");
}
.icon-wheat {
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%232d5234' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M2 22 16 8'/><path d='M3.47 12.53 5 11l1.53 1.53a3.5 3.5 0 0 1 0 4.94L5 19l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z'/><path d='M7.47 8.53 9 7l1.53 1.53a3.5 3.5 0 0 1 0 4.94L9 15l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z'/><path d='M11.47 4.53 13 3l1.53 1.53a3.5 3.5 0 0 1 0 4.94L13 11l-1.53-1.53a3.5 3.5 0 0 1 0-4.94Z'/><path d='M20 2h2v2a4 4 0 0 1-4 4h-2V6a4 4 0 0 1 4-4Z'/><path d='M11.47 17.47 13 19l-1.53 1.53a3.5 3.5 0 0 1-4.94 0L5 19l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z'/><path d='M15.47 13.47 17 15l-1.53 1.53a3.5 3.5 0 0 1-4.94 0L9 15l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z'/><path d='M19.47 9.47 21 11l-1.53 1.53a3.5 3.5 0 0 1-4.94 0L13 11l1.53-1.53a3.5 3.5 0 0 1 4.94 0Z'/></svg>");
}
.icon-arrow {
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%239a3f23' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><line x1='5' y1='12' x2='19' y2='12'/><polyline points='12 5 19 12 12 19'/></svg>");
    width: 14px;
    height: 14px;
}

@media (max-width: 768px) {
    .hero { padding: 1.8rem 1.5rem 1.5rem 1.5rem; }
    .hero-title { font-size: 1.9rem; }
    .answer-card { padding: 1.2rem 1.3rem; }
    .answer-issue { font-size: 1.3rem; }
    .brand-name { font-size: 1.4rem; }
    .topbar-session { display: none; }
    .how-grid, .impact-grid, .featured-grid { grid-template-columns: 1fr; }
    .lang-grid { grid-template-columns: repeat(2, 1fr); }
    .impact-number { font-size: 2.5rem; }
    .action-grid { grid-template-columns: 1fr; }
    .action-title { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


# ---------------- Rendering helpers ----------------
def render_answer_card(response: dict, top_score: float = 0.0, needs_escalation: bool = False, kb_sources: list = None):
    # Handle conversational responses (greetings, off-topic)
    if response.get("type") == "conversational":
        message = escape(response.get("message", ""))
        st.markdown(f'<div class="conversational-msg">{message}</div>', unsafe_allow_html=True)
        return

    # Advisory response with source separation
    if top_score >= 0.55:
        badge_class = "conf-high"
        badge_text = f"● {t('confidence_high')} · {top_score:.2f}"
    elif top_score >= 0.35:
        badge_class = "conf-medium"
        badge_text = f"● {t('confidence_medium')} · {top_score:.2f}"
    else:
        badge_class = "conf-low"
        badge_text = f"● {t('confidence_low')} · {top_score:.2f}"

    def safe(key):
        return escape(response.get(key, "") or "—")

    # Build source names string from kb_sources
    source_names = ""
    if kb_sources:
        unique_sources = list(dict.fromkeys(s.get("source", "Knowledge Base") for s in kb_sources if s.get("score", 0) > 0.3))
        if unique_sources:
            source_names = " · ".join(unique_sources[:3])

    # Build the KB-sourced sections
    html = f"""
    <div class="answer-card">
        <div class="answer-header">
            <div class="answer-kicker">AGIRITECHAT · ADVISORY</div>
            <div class="conf-badge {badge_class}">{badge_text}</div>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.8rem;">
            <div class="source-badge kb">{t('source_kb')}</div>
            {f'<div style="font-size: 0.72rem; color: var(--ink-mute); font-style: italic;">{escape(source_names)}</div>' if source_names else ''}
        </div>
        <div class="answer-issue">{safe("Likely issue")}</div>
        <div class="answer-section">
            <div class="answer-label">Why this may be happening</div>
            <div class="answer-body">{safe("Why this may be happening")}</div>
        </div>
        <div class="answer-section">
            <div class="answer-label">What to check next</div>
            <div class="answer-body">{safe("What to check next")}</div>
        </div>
        <div class="answer-section">
            <div class="answer-label">Suggested action</div>
            <div class="answer-body">{safe("Suggested action")}</div>
        </div>
        <div class="answer-section">
            <div class="answer-label">When to seek local support</div>
            <div class="answer-body">{safe("When to seek local support")}</div>
        </div>
    """

    # Add AI additional context section if present and non-trivial
    ai_context = response.get("AI additional context", "")
    if ai_context and ai_context.lower().strip() not in ("", "—", "no additional context needed.", "no additional context needed"):
        html += f"""
        <div class="ai-context-section">
            <div class="source-badge ai">{t('source_ai')}</div>
            <div style="font-size: 0.72rem; color: var(--ink-mute); font-style: italic; margin-bottom: 0.4rem;">Generated by Llama 3.3 70B via Groq — not from verified knowledge base</div>
            <div class="answer-body">{escape(ai_context)}</div>
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    if needs_escalation:
        st.markdown(
            f'<div class="escalate-box">⚠ {t("escalation")}</div>',
            unsafe_allow_html=True,
        )


def process_question(user_question, crop_hint, category_hint="general",
                     image_symptoms=None, image_source="none"):
    with st.spinner("…"):
        state = run_agent(
            user_question=user_question,
            crop_hint=crop_hint,
            category_hint=category_hint,
            image_symptoms=image_symptoms or [],
            image_source=image_source,
            language=st.session_state.language,
            farmer_profile=st.session_state.farmer_profile,
        )

    interaction_id = log_interaction(st.session_state.session_id, state)

    render_answer_card(
        state.get("response", {}),
        state.get("top_score", 0.0),
        state.get("needs_escalation", False),
        kb_sources=state.get("kb_sources", []),
    )

    # Only show feedback buttons for advisory responses
    response_type = (state.get("response") or {}).get("type", "advisory")
    if response_type == "advisory":
        fb1, fb2, _ = st.columns([1, 1, 8])
        if fb1.button("👍 " + t("helpful"), key=f"up_{interaction_id}"):
            record_feedback(interaction_id, 1)
            st.toast(t("thanks_feedback"))
        if fb2.button("👎 " + t("not_helpful"), key=f"down_{interaction_id}"):
            record_feedback(interaction_id, -1)
            st.toast(t("thanks_feedback"))

    # Show KB sources with attribution — always visible for advisory
    kb_sources = state.get("kb_sources", [])
    matches = state.get("matches", [])
    sources_to_show = kb_sources if kb_sources else matches
    if sources_to_show and response_type == "advisory":
        with st.expander(f"📚 {t('sources_used')} ({len(sources_to_show)})"):
            for m in sources_to_show:
                source_name = m.get("source", "Knowledge Base")
                score_val = m.get("score", 0)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;">'
                    f'<div class="source-badge kb">{escape(source_name)}</div>'
                    f'<span style="font-size:0.72rem;color:var(--ink-mute);">Score: {score_val:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{m['question']}**")
                st.write(m.get("answer", ""))
                st.write("---")

    # Agent trace only shown in developer view
    if st.session_state.developer_view:
        with st.expander(f"⚙ {t('agent_trace')}"):
            st.write("**Path:**", " → ".join(state.get("trace", [])))
            st.write("**Language:**", LANGUAGES[state.get("language", "en")]["name"])
            st.write("**Classified crop:**", state.get("classified_crop", "unknown"))
            st.write("**Is agricultural:**", state.get("is_agricultural", True))
            st.write("**Top retrieval score:**", round(state.get("top_score", 0.0), 3))
            if state.get("image_source") and state.get("image_source") != "none":
                st.write("**Image analysis source:**", state.get("image_source"))

    st.session_state.messages.append({
        "role": "assistant",
        "response": state.get("response"),
        "top_score": state.get("top_score", 0.0),
        "needs_escalation": state.get("needs_escalation", False),
        "kb_sources": state.get("kb_sources", []),
        "interaction_id": interaction_id,
    })


# ---------------- Sidebar ----------------
with st.sidebar:
    # Language selector
    lang_names = [LANGUAGES[code]["name"] for code in ["en", "sw", "fr", "rw"]]
    lang_codes = ["en", "sw", "fr", "rw"]
    current_idx = lang_codes.index(st.session_state.language)
    st.markdown(f'<div style="font-family: \'DM Sans\', sans-serif; font-size: 0.78rem; font-weight: 700; color: var(--forest-700); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;"><span class="icon-globe"></span>{t("sidebar_language")}</div>', unsafe_allow_html=True)
    selected_lang = st.selectbox(
        "language_selector",
        options=lang_names,
        index=current_idx,
        label_visibility="collapsed",
    )
    new_lang = lang_codes[lang_names.index(selected_lang)]
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()

    st.markdown("---")

    # Farmer profile
    st.markdown(f"### {t('sidebar_profile')}")

    name = st.text_input(t("name"), value=st.session_state.farmer_profile.get("name", ""), key="f_name")
    region = st.text_input(t("region"), value=st.session_state.farmer_profile.get("region", ""), key="f_region", placeholder="e.g. Musanze, Kigali…")
    farm_size = st.text_input(t("farm_size"), value=st.session_state.farmer_profile.get("farm_size", ""), key="f_size", placeholder="e.g. 0.5 ha")
    crops_grown = st.text_input(t("crops"), value=st.session_state.farmer_profile.get("crops", ""), key="f_crops", placeholder="e.g. Maize, Soybean")
    planting_date = st.text_input(t("planting_date"), value=st.session_state.farmer_profile.get("planting_date", ""), key="f_planting", placeholder="e.g. March 2026")

    if st.button(t("save_profile"), use_container_width=True):
        st.session_state.farmer_profile = {
            "name": name.strip(),
            "region": region.strip(),
            "farm_size": farm_size.strip(),
            "crops": crops_grown.strip(),
            "planting_date": planting_date.strip(),
        }
        st.session_state.profile_saved = True
        st.toast(t("profile_saved"))

    st.caption(t("profile_note"))

    st.markdown("---")

    # Crop & topic focus
    st.markdown('<div style="font-family: \'DM Sans\', sans-serif; font-size: 0.78rem; font-weight: 700; color: var(--forest-700); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem;"><span class="icon-wheat"></span>Crop focus</div>', unsafe_allow_html=True)
    selected_crop = st.selectbox(
        "crop_focus",
        ["General", "Maize", "Soybean"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Developer view toggle (new)
    dev_view = st.checkbox(
        t('sidebar_dev'),
        value=st.session_state.developer_view,
        help=t("sidebar_dev_help"),
    )
    if dev_view != st.session_state.developer_view:
        st.session_state.developer_view = dev_view
        st.rerun()

    # Status — only visible in developer view
    if st.session_state.developer_view:
        st.markdown("---")
        st.markdown(f"**{t('sidebar_status')}**")
        if llm_available():
            st.write("🟢 AI reasoning: active")
        else:
            st.write("🟡 AI reasoning: offline")
        st.write("🟢 Semantic retrieval")
        st.write("🟢 Image analysis")
        st.write("🟢 Feedback logging")

        with st.expander(f"📊 {t('feedback_stats')}"):
            stats = recent_stats()
            st.write(f"Total interactions: **{stats['total']}**")
            st.write(f"👍 {stats['thumbs_up']}   👎 {stats['thumbs_down']}")
            st.write(f"Escalations: **{stats['escalations']}**")


# ---------------- Top bar ----------------
session_badge = (
    f'<div class="topbar-session">Session · {st.session_state.session_id}</div>'
    if st.session_state.developer_view else ""
)
st.markdown(f"""
<div class="topbar">
    <div class="brand-group">
        <div class="brand-mark">A</div>
        <div>
            <div class="brand-name">AgiriteChat</div>
            <div class="brand-tag">{t('brand_tag')}</div>
        </div>
    </div>
    {session_badge}
</div>
""", unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown(f"""
<div class="hero">
    <div class="hero-label">AgiriteChat · v3</div>
    <div class="hero-title">{t('hero_title')}</div>
    <div class="hero-sub">{t('hero_sub')}</div>
</div>
""", unsafe_allow_html=True)

# ---------------- Category filter buttons (real working filters) ----------------
st.markdown(f'<div class="category-label">QUICK FILTERS</div>', unsafe_allow_html=True)

fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1.3, 5])
if fcol1.button(t("filter_pests"),
                use_container_width=True,
                type="primary" if st.session_state.category_filter == "pest" else "secondary"):
    st.session_state.category_filter = None if st.session_state.category_filter == "pest" else "pest"
    st.rerun()
if fcol2.button(t("filter_diseases"),
                use_container_width=True,
                type="primary" if st.session_state.category_filter == "disease" else "secondary"):
    st.session_state.category_filter = None if st.session_state.category_filter == "disease" else "disease"
    st.rerun()
if fcol3.button(t("filter_soil"),
                use_container_width=True,
                type="primary" if st.session_state.category_filter == "soil" else "secondary"):
    st.session_state.category_filter = None if st.session_state.category_filter == "soil" else "soil"
    st.rerun()

# Active filter badge
if st.session_state.category_filter:
    filter_label_key = f"filter_{st.session_state.category_filter}s" if st.session_state.category_filter != "soil" else "filter_soil"
    filter_display = t(filter_label_key)
    bc1, bc2 = st.columns([3, 9])
    with bc1:
        st.markdown(
            f'<div class="active-filter-badge">● {t("filter_active")}: {filter_display}</div>',
            unsafe_allow_html=True,
        )
    with bc2:
        if st.button(f"✕ {t('filter_clear')}", key="clear_filter"):
            st.session_state.category_filter = None
            st.rerun()

# ============================================================
# MAIN INTERACTION AREA — Ask / Photo / Library
# Farmers need these immediately. Big tabs, no decoration.
# ============================================================

tab1, tab2, tab3 = st.tabs([
    f"💬  {t('action_ask_title')}",
    f"📷  {t('action_photo_title')}",
    f"📚  {t('tab_browse')}",
])

# ---- ASK A QUESTION ----
with tab1:
    # Welcome card
    profile = st.session_state.farmer_profile
    has_profile = st.session_state.profile_saved and profile.get("name")
    if has_profile:
        avatar = profile["name"][0].upper() if profile["name"] else "F"
        welcome = t("welcome_named").format(name=profile["name"])
        meta_parts = [p for p in [profile.get("region"), profile.get("crops"), profile.get("farm_size")] if p]
        meta = " · ".join(meta_parts) if meta_parts else ""
        meta_html = f'<div class="welcome-meta">{escape(meta)}</div>' if meta else ""
        st.markdown(
            f'<div class="welcome-card">'
            f'<div class="welcome-avatar">{escape(avatar)}</div>'
            f'<div><div class="welcome-text">{escape(welcome)}</div>{meta_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Quick starts
    crop_key = selected_crop.lower() if selected_crop != "General" else "general"
    lang_key = st.session_state.language
    preset_list = PRESETS.get(crop_key, PRESETS["general"]).get(lang_key, PRESETS[crop_key]["en"])

    if st.session_state.category_filter:
        preset_list = [p for p in preset_list if p[2] == st.session_state.category_filter]

    if preset_list:
        st.markdown(f'<div class="section-header">{t("quick_starts")}</div>', unsafe_allow_html=True)
        num_cols = min(len(preset_list), 4)
        cols = st.columns(num_cols)
        for i, (label, question, _cat) in enumerate(preset_list):
            if cols[i % num_cols].button(label, key=f"preset_{i}_{lang_key}_{crop_key}", use_container_width=True):
                st.session_state["preset_q"] = question

    # Message history
    for msg in st.session_state.messages:
        if msg.get("response"):
            render_answer_card(msg["response"], msg.get("top_score", 0.0), msg.get("needs_escalation", False), kb_sources=msg.get("kb_sources", []))

    preset_q = st.session_state.pop("preset_q", None)
    if preset_q:
        with st.chat_message("user"):
            st.write(preset_q)
        with st.chat_message("assistant"):
            process_question(preset_q, selected_crop)

    user_q = st.chat_input(t("input_placeholder"))
    if user_q:
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            process_question(user_q, selected_crop)

# ---- UPLOAD A PHOTO ----
with tab2:
    st.markdown(f'<div class="section-sub" style="margin-bottom: 0.8rem;">{t("photo_upload")}</div>', unsafe_allow_html=True)

    photo = st.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="photo_uploader_main")
    photo_desc = st.text_input(t("photo_desc"), placeholder=t("photo_desc_ph"), key="photo_desc_main")

    if photo is None:
        st.markdown("""
        <div style="
            background: linear-gradient(180deg, var(--forest-100) 0%, var(--cream) 100%);
            border: 1px dashed #c8dcc6; border-radius: 16px;
            padding: 2.5rem 1.5rem; text-align: center; margin: 1rem 0;
        ">
            <div style="width:110px;height:110px;margin:0 auto 1rem;border-radius:50%;
                background-image:url('https://images.pexels.com/photos/20111827/pexels-photo-20111827.jpeg?auto=compress&cs=tinysrgb&w=240&h=240&fit=crop');
                background-size:cover;background-position:center;border:3px solid var(--cream);box-shadow:var(--shadow-warm);"></div>
            <div style="font-family:'Fraunces',serif;font-size:1.15rem;font-weight:600;color:var(--forest-900);margin-bottom:0.4rem;">No photo uploaded yet</div>
            <div style="font-size:0.9rem;color:var(--ink-soft);max-width:360px;margin:0 auto;line-height:1.5;">Upload a close-up photo of an affected leaf or plant. Best results come from clear, well-lit photos in daylight.</div>
        </div>
        """, unsafe_allow_html=True)

    if photo is not None:
        st.image(photo, use_container_width=True)
        if st.button(t("analyze_photo"), type="primary", use_container_width=True, key="analyze_btn_main"):
            from vision import analyze_photo
            image_bytes = photo.getvalue()
            with st.spinner("Analyzing photo…"):
                vision_result = analyze_photo(image_bytes)
            if vision_result and vision_result.get("symptoms"):
                st.markdown(f'<div class="section-header">{t("symptoms_detected")}</div>', unsafe_allow_html=True)
                for sym in vision_result["symptoms"]:
                    st.write(f"• {sym}")
                question = photo_desc if photo_desc else "What is wrong with this crop?"
                process_question(question, selected_crop, image_symptoms=vision_result["symptoms"], image_source=vision_result.get("source", "groq_vision"))
            elif vision_result and vision_result.get("error"):
                st.warning(vision_result["error"])
            else:
                st.warning("Could not analyze the photo. Please try a clearer image.")

# ---- KNOWLEDGE LIBRARY ----
with tab3:
    search_term = st.text_input(t("library_search"), placeholder=t("library_search_ph"), label_visibility="collapsed", key="lib_search_main")
    retriever = get_retriever()
    if search_term:
        hits = retriever.search(query=search_term, top_k=8)
        if st.session_state.category_filter:
            target_cats = CATEGORY_MAP.get(st.session_state.category_filter, [])
            hits = [h for h in hits if h.get("category") in target_cats]
        for h in hits:
            with st.expander(h["question"]):
                st.write(h["answer"])
                st.caption(f"{h.get('crop', '')} · {h.get('category', '')} · Source: {h.get('source', 'Knowledge Base')}")
    else:
        st.info("Search the knowledge base by typing symptoms or topics above.")

# ============================================================
# LANDING SECTIONS — below the interaction area
# ============================================================

# ---- Section 1: How it works ----
st.markdown(f"""
<div class="land-section">
    <div class="land-kicker">{t('how_kicker')}</div>
    <div class="land-title">{t('how_title')}</div>
    <div class="how-grid">
        <div class="how-card">
            <div class="how-card-image" style="background-image: url('https://images.pexels.com/photos/9324755/pexels-photo-9324755.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop');"></div>
            <div class="how-step-number">STEP 01</div>
            <div class="how-step-title">{t('how_step1_title')}</div>
            <div class="how-step-desc">{t('how_step1_desc')}</div>
        </div>
        <div class="how-card">
            <div class="how-card-image" style="background-image: url('https://images.pexels.com/photos/28301257/pexels-photo-28301257.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop');"></div>
            <div class="how-step-number">STEP 02</div>
            <div class="how-step-title">{t('how_step2_title')}</div>
            <div class="how-step-desc">{t('how_step2_desc')}</div>
        </div>
        <div class="how-card">
            <div class="how-card-image" style="background-image: url('https://images.pexels.com/photos/20111827/pexels-photo-20111827.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop');"></div>
            <div class="how-step-number">STEP 03</div>
            <div class="how-step-title">{t('how_step3_title')}</div>
            <div class="how-step-desc">{t('how_step3_desc')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Section 2: Real impact (live stats from feedback DB) ----
try:
    _stats = recent_stats()
    _total = _stats.get('total', 0)
    _escalations = _stats.get('escalations', 0)
    _confident = max(0, _total - _escalations)
except Exception:
    _total = 0
    _confident = 0

st.markdown(f"""
<div class="land-section">
    <div class="land-kicker">{t('impact_kicker')}</div>
    <div class="land-title">{t('impact_title')}</div>
    <div class="impact-grid">
        <div class="impact-card">
            <div class="impact-number">{_total}</div>
            <div class="impact-label">{t('impact_questions')}</div>
        </div>
        <div class="impact-card terracotta">
            <div class="impact-number">{_confident}</div>
            <div class="impact-label">{t('impact_confident')}</div>
        </div>
        <div class="impact-card gold">
            <div class="impact-number">4</div>
            <div class="impact-label">{t('impact_languages')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- Section 3: Featured topics (real KB cards, clickable) ----
# Pull three real entries from the knowledge base
import json as _json
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as _f:
        _kb_all = _json.load(_f)
except Exception:
    _kb_all = []

def _find_kb(question_substring, fallback_question):
    """Find a KB entry whose question contains the substring; fallback to a literal Q."""
    for e in _kb_all:
        if question_substring.lower() in e.get("question", "").lower():
            return e
    return {"question": fallback_question, "answer": "", "crop": "general", "category": "general"}

_featured = [
    {
        "entry": _find_kb("fall armyworm", "How do I control fall armyworm in maize?"),
        "tag": "Maize · Pest",
        "image": "https://images.pexels.com/photos/9324755/pexels-photo-9324755.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop",
    },
    {
        "entry": _find_kb("purple", "My maize leaves are turning purple. What is wrong?"),
        "tag": "Maize · Soil",
        "image": "https://images.pexels.com/photos/13525130/pexels-photo-13525130.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop",
    },
    {
        "entry": _find_kb("nodulation", "How can I improve soybean nodulation?"),
        "tag": "Soybean · Soil",
        "image": "https://images.pexels.com/photos/28301257/pexels-photo-28301257.jpeg?auto=compress&cs=tinysrgb&w=600&h=400&fit=crop",
    },
]

# Build all card HTML first, then render as one block.
# Avoid html.escape on card content because it was causing double-escaping issues
# that made Streamlit render raw HTML tags as text.
_cards_html_parts = []
for i, _f in enumerate(_featured):
    _e = _f["entry"]
    _q = (_e.get("question") or "").replace("<", "&lt;").replace(">", "&gt;")
    _raw_answer = (_e.get("answer") or "")
    _summary = _raw_answer[:140].replace("<", "&lt;").replace(">", "&gt;")
    if len(_raw_answer) > 140:
        _summary += "…"
    _tag = _f["tag"].replace("<", "&lt;").replace(">", "&gt;")
    _img = _f["image"]
    _cta = t('featured_card_open')
    _cards_html_parts.append(
        '<div class="featured-card">'
        f'<div class="featured-image" style="background-image: url(\'{_img}\');">'
        f'<div class="featured-tag">{_tag}</div>'
        '</div>'
        '<div class="featured-body">'
        f'<div class="featured-title">{_q}</div>'
        f'<div class="featured-summary">{_summary}</div>'
        f'<div class="featured-cta">{_cta} <span class="icon-arrow"></span></div>'
        '</div>'
        '</div>'
    )

_all_cards = "".join(_cards_html_parts)
_kicker = t('featured_kicker')
_title = t('featured_title')

st.markdown(
    f'<div class="land-section">'
    f'<div class="land-kicker">{_kicker}</div>'
    f'<div class="land-title">{_title}</div>'
    f'<div class="featured-grid">{_all_cards}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Wire up the cards: real Streamlit buttons sit just below as click triggers
_btn_cols = st.columns(3)
for i, _f in enumerate(_featured):
    with _btn_cols[i]:
        if st.button(t('featured_card_open'), key=f"featured_btn_{i}", use_container_width=True):
            st.session_state["preset_q"] = _f["entry"]["question"]
            st.rerun()

# ---- Section 4: Languages strip ----
st.markdown(f"""
<div class="lang-section">
    <div class="land-kicker">{t('languages_kicker')}</div>
    <div class="land-title" style="margin-bottom: 0.4rem;">{t('languages_title')}</div>
    <div style="font-size: 0.95rem; color: var(--ink-soft); max-width: 580px; line-height: 1.55;">{t('languages_sub')}</div>
    <div class="lang-grid">
        <div class="lang-pill">
            <div class="lang-pill-name">English</div>
            <div class="lang-pill-native">EN</div>
        </div>
        <div class="lang-pill">
            <div class="lang-pill-name">Kiswahili</div>
            <div class="lang-pill-native">SW</div>
        </div>
        <div class="lang-pill">
            <div class="lang-pill-name">Français</div>
            <div class="lang-pill-native">FR</div>
        </div>
        <div class="lang-pill beta">
            <div class="lang-pill-name">Kinyarwanda</div>
            <div class="lang-pill-native">RW</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown(f"""
<div class="footer-card">
    <strong>{t('responsible_use')}</strong>
    {t('responsible_text')}
</div>
""", unsafe_allow_html=True)
