"""
web_app.py  —  IDA AI Screening Tool (Polished Dashboard Edition)
====================================================================
Premium, app-store-quality interface:
  - Card-based layout, icons, smooth visual hierarchy
  - Crisp outcome: "Iron Deficiency: Present / Absent"
  - Hindi / English toggle
  - Patient View (simple) vs Doctor View (full analytics)
  - Confidence gauge, symptom badges, patient explanation chart

Run train_model.py first, then:  python -m streamlit run web_app.py
"""

import os, io, json, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import joblib, pandas as pd, numpy as np
import shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="IDA Screening | आयरन जांच", page_icon="assets/logo_32.png", layout="centered")

def _logo_data_uri():
    """Base64-embeds the app logo so it can be dropped straight into the
    hero HTML block (avoids Streamlit's separate image-loading round trip)."""
    import base64
    try:
        with open(os.path.join(os.path.dirname(__file__), "assets", "logo_64.png"), "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return None
_LOGO_URI = _logo_data_uri()
_logo_html = f'<img src="{_LOGO_URI}" alt="logo">' if _LOGO_URI else "🩸"

# ============================================================================
# STYLE
# ============================================================================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem; max-width: 760px;}

/* Hard-force the dark theme at the top level, regardless of the browser's
   light/dark preference or any client-side override — this whole stylesheet
   is designed for dark backgrounds (light text colors), so if light mode
   ever slips through, text becomes low-contrast/hard to read ("blurry"). */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stSidebar"], [data-testid="stForm"] {
    background-color: #0e0e16 !important;
    color: #e8e8f0 !important;
}
.stApp, .stApp * { color-scheme: dark !important; }

.brandbar { display:flex; align-items:center; gap:10px; margin:2px 0 14px 0; }
.brandbar img { height:34px; width:auto; display:block; filter:drop-shadow(0 2px 5px rgba(0,0,0,0.35)); }
.brandbar span { color:#e8e8f0; font-size:14px; font-weight:700; letter-spacing:.02em; }

.hero {
    background: linear-gradient(120deg, #0F6E56 0%, #14856A 55%, #1a9e80 100%);
    border-radius: 18px; padding: 26px 28px; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(15,110,86,0.25);
}
.hero-row { display:flex; align-items:center; gap:16px; }
.hero-icon { font-size:38px; line-height:1; }
.hero-icon img { height:44px; width:auto; display:block; filter:drop-shadow(0 2px 6px rgba(0,0,0,0.25)); }
.hero h1 { color:white; font-size:21px; font-weight:700; margin:0; }
.hero p  { color:rgba(255,255,255,0.85); font-size:13px; margin:5px 0 0 0; }

.stat-grid { display:flex; gap:10px; margin-bottom:18px; }
.stat-card {
    flex:1; background:var(--background-color,#1c1c2b); border:1px solid rgba(255,255,255,0.08);
    border-radius:14px; padding:14px 16px; position:relative; overflow:hidden;
}
.stat-card::before {
    content:""; position:absolute; top:0; left:0; width:4px; height:100%;
    background:linear-gradient(180deg,#1D9E75,#0F6E56);
}
.stat-icon { font-size:18px; margin-bottom:6px; opacity:0.85; }
.stat-label { font-size:11px; color:#b4b4c8; font-weight:600; letter-spacing:.02em; }
.stat-value { font-size:24px; font-weight:800; color:#f2f2f5; margin:3px 0 1px 0; }
.stat-sub { font-size:10.5px; color:#8a8aa0; font-weight:500; }

.section-card {
    background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
    border-radius:16px; padding:18px 20px 8px 20px; margin-bottom:16px;
}
.section-title { display:flex; align-items:center; gap:8px; font-size:13px; font-weight:700;
    color:#dcdcec; text-transform:uppercase; letter-spacing:.06em; margin-bottom:14px; }
.section-title .icon { font-size:16px; }

.toprow { display:flex; justify-content:flex-end; gap:8px; margin-bottom:10px; }

.result-box {
    border-radius:18px; padding:22px 24px; margin-top:18px;
    display:flex; align-items:center; gap:18px;
    animation: fadeIn .35s ease-in;
}
@keyframes fadeIn { from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:translateY(0);} }
.result-box.present { background:linear-gradient(120deg,#2d1212,#3a1414); border:1px solid #c0392b55; }
.result-box.absent  { background:linear-gradient(120deg,#0e2a1c,#0f3322); border:1px solid #27ae6055; }
.result-icon-wrap {
    width:56px; height:56px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:26px; flex-shrink:0;
}
.result-box.present .result-icon-wrap { background:rgba(231,76,60,0.18); }
.result-box.absent  .result-icon-wrap { background:rgba(46,204,113,0.18); }
.result-title { font-size:19px; font-weight:800; margin:0; }
.result-box.present .result-title { color:#ff6b5b; }
.result-box.absent  .result-title { color:#3ddc84; }
.result-advice { font-size:12.5px; color:#cacad8; font-weight:500; margin-top:6px; }

.gauge-wrap { margin-top:10px; }
.gauge-track { width:100%; height:8px; border-radius:99px; background:rgba(255,255,255,0.08); overflow:hidden; }
.gauge-fill  { height:100%; border-radius:99px; transition: width .6s ease; }
.gauge-fill.present { background:linear-gradient(90deg,#e74c3c,#ff8a75); }
.gauge-fill.absent  { background:linear-gradient(90deg,#27ae60,#5be08c); }
.gauge-label { font-size:11px; color:#a8a8ba; font-weight:500; margin-top:5px; }

.disclaimer-box {
    text-align:center; font-size:11.5px; color:#8a8aa0; font-weight:500; margin-top:26px;
    padding:12px; border-top:1px solid rgba(255,255,255,0.07);
}

div[data-testid="stForm"] { border:none; padding:0; }
.stButton button {
    border-radius:12px !important; font-weight:700 !important;
}

/* Fix: selectbox was truncating long bilingual options (Hindi + English)
   with "..." so the English part was hidden. Let it wrap fully. */
div[data-baseweb="select"] > div {
    height:auto !important;
    min-height:44px;
}
div[data-baseweb="select"] span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height:1.35;
}
ul[data-testid="stSelectboxVirtualDropdown"] li,
div[role="listbox"] li {
    white-space: normal !important;
    line-height:1.35;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    for p in ["ida_model.joblib", "feature_list.joblib"]:
        if not os.path.exists(p):
            st.error(f"Missing {p} — run train_model.py first.")
            st.stop()
    m      = joblib.load("ida_model.joblib")
    feats  = joblib.load("feature_list.joblib")
    scaler = joblib.load("scaler.joblib")     if os.path.exists("scaler.joblib")     else None
    meta   = joblib.load("model_meta.joblib") if os.path.exists("model_meta.joblib") else {"name":"Model","uses_scaled_input":False}
    return m, feats, scaler, meta

model, FEATURE_COLUMNS, scaler, model_meta = load_model()
USES_SCALED = model_meta.get("uses_scaled_input", False)

# ============================================================================
# PATIENT DATA COLLECTION — Cloud database (Supabase/Postgres), with a
# local SQLite fallback for testing without any setup.
#
# (per Dr. Ruchita Dixit's research protocol: real-time patient enrollment,
# recording demographics + symptoms + lab-confirmed IDA status, for
# statistical analysis and future model retraining)
#
# HOW THIS WORKS:
#   - If a "SUPABASE_DB_URL" secret is configured (see README for setup),
#     all patient records are saved to that real, permanent cloud database
#     — this is what should be used for real hospital/study use, since it
#     survives app restarts, redeploys, and sleep/wake cycles.
#   - If no such secret is configured, it automatically falls back to a
#     local SQLite file (patient_records.db) — convenient for local
#     testing, but NOT persistent if deployed on Streamlit Community Cloud.
# ============================================================================
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float

def _clean_db_url(raw):
    """Fixes the most common paste mistakes in the connection string:
    surrounding quotes/spaces/newlines and the old 'postgres://' prefix."""
    url = str(raw).strip().strip('"').strip("'").strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url

@st.cache_resource(show_spinner=False)
def _try_cloud_engine():
    """Connects to the cloud (Supabase) database, retrying a few times because
    the first connection from Streamlit Cloud can be slow. Failures are NOT
    cached (Streamlit only caches successful returns), so the next page load
    simply tries again instead of being stuck on local storage."""
    import time
    db_url = _clean_db_url(st.secrets["SUPABASE_DB_URL"])
    last_err = None
    for attempt in range(3):
        try:
            engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_recycle=300,
                connect_args={
                    "connect_timeout": 20,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                },
            )
            with engine.connect():
                pass
            return engine
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err

def get_db_engine():
    try:
        return _try_cloud_engine(), True
    except Exception as e:
        # Remember the real reason so it can be shown under the warning banner.
        st.session_state["db_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return create_engine("sqlite:///patient_records.db"), False

_metadata = MetaData()
_patient_columns = (
    [Column("record_id", Integer, primary_key=True, autoincrement=True),
     Column("patient_id", String),
     Column("patient_name", String),
     Column("patient_phone", String),
     Column("entry_timestamp", String)]
    + [Column(c, Integer) for c in FEATURE_COLUMNS]
    + [Column("hemoglobin", Float),
       Column("mcv", Float),
       Column("mch", Float),
       Column("mchc", Float),
       Column("rdwcv", Float),
       Column("lab_panel_json", String),
       Column("lab_confirmed_ida", String),
       Column("ai_predicted_ida", String),
       Column("ai_confidence", Float),
       Column("kuppuswamy_score", Integer),
       Column("kuppuswamy_class", String),
       Column("clinician_assessment", String),
       Column("clinician_notes", String),
       Column("consent_given", Integer),
       Column("is_active", Integer, default=1)]
)
patients_table = Table("patients", _metadata, *_patient_columns)

def ensure_is_active_column(engine):
    """Adds the is_active column to an already-existing patients table that
    was created before this feature existed (safe migration, runs once)."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    if "is_active" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE patients ADD COLUMN is_active INTEGER DEFAULT 1"))
            conn.execute(text("UPDATE patients SET is_active = 1 WHERE is_active IS NULL"))

def ensure_kuppuswamy_columns(engine):
    """Adds kuppuswamy_score / kuppuswamy_class to an already-existing
    patients table created before this feature existed (safe migration)."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    with engine.begin() as conn:
        if "kuppuswamy_score" not in cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN kuppuswamy_score INTEGER"))
        if "kuppuswamy_class" not in cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN kuppuswamy_class VARCHAR"))

def ensure_clinician_columns(engine):
    """Adds patient_phone / clinician_assessment / clinician_notes to an
    already-existing patients table (safe migration, per Dr. Dixit's
    modifications #1 and #5)."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    with engine.begin() as conn:
        if "patient_phone" not in cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN patient_phone VARCHAR"))
        if "clinician_assessment" not in cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN clinician_assessment VARCHAR"))
        if "clinician_notes" not in cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN clinician_notes VARCHAR"))

def ensure_lab_panel_column(engine):
    """Adds lab_panel_json to an already-existing patients table (safe
    migration, for the expanded CBC/PS lab panel)."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    if "lab_panel_json" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE patients ADD COLUMN lab_panel_json VARCHAR"))

def ensure_cbc_core_columns(engine):
    """Adds mcv/mch/mchc/rdwcv quick-access columns to an already-existing
    patients table (safe migration). These replaced the old Iron Profile
    quick-access columns (serum_ferritin/serum_iron/tibc/transferrin_saturation)
    once the Iron Profile was removed from the diagnostic calculation —
    the core IDA verdict is now based on CBC + Peripheral Smear only."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    with engine.begin() as conn:
        for col in ("mcv", "mch", "mchc", "rdwcv"):
            if col not in cols:
                conn.execute(text(f"ALTER TABLE patients ADD COLUMN {col} FLOAT"))

def ensure_consent_column(engine):
    """Adds consent_given to an already-existing patients table (safe
    migration) — records whether patient/guardian consent was obtained
    before their symptoms and health data were collected."""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    if "consent_given" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE patients ADD COLUMN consent_given INTEGER"))

def init_db(engine):
    _metadata.create_all(engine, checkfirst=True)
    ensure_is_active_column(engine)
    ensure_kuppuswamy_columns(engine)
    ensure_clinician_columns(engine)
    ensure_lab_panel_column(engine)
    ensure_consent_column(engine)
    ensure_cbc_core_columns(engine)

def _to_native(v):
    """Convert numpy/pandas scalar types to native Python types so the DB
    driver (especially psycopg2 for Postgres) can serialize them correctly."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

def save_patient_record(patient_id, patient_name, symptom_values, lab_values, lab_confirmed, ai_pred, ai_conf,
                         kup_score=None, kup_class=None, patient_phone=None,
                         clinician_assessment=None, clinician_notes=None, consent_given=False):
    engine, _ = get_db_engine()
    init_db(engine)
    clean_lab = {k: _to_native(v) for k, v in lab_values.items() if v is not None and v != ""}
    row = {
        "patient_id": patient_id, "patient_name": patient_name, "patient_phone": patient_phone,
        "entry_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{c: _to_native(symptom_values[c]) for c in FEATURE_COLUMNS},
        "hemoglobin": clean_lab.get("hb"), "mcv": clean_lab.get("mcv"),
        "mch": clean_lab.get("mch"), "mchc": clean_lab.get("mchc"),
        "rdwcv": clean_lab.get("rdwcv"),
        "lab_panel_json": json.dumps(clean_lab),
        "lab_confirmed_ida": lab_confirmed, "ai_predicted_ida": ai_pred, "ai_confidence": _to_native(ai_conf),
        "kuppuswamy_score": _to_native(kup_score) if kup_score is not None else None,
        "kuppuswamy_class": kup_class,
        "clinician_assessment": clinician_assessment, "clinician_notes": clinician_notes,
        "consent_given": 1 if consent_given else 0,
        "is_active": 1,
    }
    with engine.begin() as conn:
        conn.execute(patients_table.insert().values(**row))

def patient_id_exists(patient_id):
    """Returns the list of existing active records with this exact
    Patient ID (used to warn staff before creating an accidental
    duplicate)."""
    engine, _ = get_db_engine()
    init_db(engine)
    with engine.connect() as conn:
        rows = conn.execute(
            patients_table.select().where(
                (patients_table.c.patient_id == patient_id) & (patients_table.c.is_active == 1)
            )
        ).fetchall()
    return rows

def load_all_records(active_only=True):
    engine, _ = get_db_engine()
    init_db(engine)
    with engine.connect() as conn:
        query = patients_table.select().order_by(patients_table.c.record_id.desc())
        query = query.where(patients_table.c.is_active == (1 if active_only else 0))
        df = pd.read_sql_query(query, conn)
    return df

def hide_record(record_id):
    engine, _ = get_db_engine()
    with engine.begin() as conn:
        conn.execute(patients_table.update().where(patients_table.c.record_id == record_id).values(is_active=0))

def restore_record(record_id):
    engine, _ = get_db_engine()
    with engine.begin() as conn:
        conn.execute(patients_table.update().where(patients_table.c.record_id == record_id).values(is_active=1))

def delete_record_permanently(record_id):
    engine, _ = get_db_engine()
    with engine.begin() as conn:
        conn.execute(patients_table.delete().where(patients_table.c.record_id == record_id))

# ============================================================================
# LANGUAGE
# ============================================================================
LANG = {
"en": {
    "title":"IDA Screening", "subtitle":"AI diagnostic tool · Signs & symptoms only · No blood test required",
    "lang_btn":"🇮🇳 हिंदी", "view_doc":"🩺 Doctor view", "view_pat":"👤 Patient view", "data_collection":"Data Collection",
    "acc":"Accuracy","auc":"AUC-ROC","data":"Training data",
    "demo":"Patient information","symp":"Signs & symptoms","btn":"Run AI prediction",
    "present":"Iron Deficiency: Present", "absent":"Iron Deficiency: Absent",
    "conf":"Confidence", "advice_p":"Consult a doctor and confirm with a blood test.",
    "advice_a":"No iron deficiency indicated. See a doctor if symptoms persist.",
    "disclaimer":"Screening aid only — not a substitute for clinical diagnosis. Built from a research dataset for academic purposes.",
    "why":"Why this result", "why_cap":"Red pushes toward IDA · Green pushes away · Longer bar = stronger effect",
    "fi":"Feature importance","shap":"SHAP summary","cmp":"Model comparison",
    "age":"Age (years)","gender":"Gender","address":"Residence","edu":"Education","occ":"Occupation","income":"Monthly income (₹)",
},
"hi": {
    "title":"IDA जांच", "subtitle":"AI निदान उपकरण · केवल लक्षणों पर आधारित · रक्त परीक्षण आवश्यक नहीं",
    "lang_btn":"🇬🇧 English", "view_doc":"🩺 डॉक्टर दृश्य", "view_pat":"👤 मरीज़ दृश्य", "data_collection":"डेटा संग्रह",
    "acc":"सटीकता","auc":"AUC-ROC","data":"प्रशिक्षण डेटा",
    "demo":"मरीज़ की जानकारी","symp":"संकेत और लक्षण","btn":"AI जांच करें",
    "present":"आयरन की कमी: उपस्थित", "absent":"आयरन की कमी: अनुपस्थित",
    "conf":"विश्वास", "advice_p":"कृपया डॉक्टर से मिलें और रक्त परीक्षण से पुष्टि करें।",
    "advice_a":"आयरन की कमी नहीं दिखती। लक्षण बने रहने पर डॉक्टर से मिलें।",
    "disclaimer":"यह केवल जांच सहायक है — चिकित्सीय निदान का विकल्प नहीं। शोध डेटा पर आधारित शैक्षणिक परियोजना।",
    "why":"यह परिणाम क्यों", "why_cap":"लाल = IDA की ओर · हरा = IDA से दूर · लंबा बार = अधिक प्रभाव",
    "fi":"फीचर महत्व","shap":"SHAP सारांश","cmp":"मॉडल तुलना",
    "age":"आयु (वर्ष)","gender":"लिंग","address":"निवास","edu":"शिक्षा","occ":"व्यवसाय","income":"मासिक आय (₹)",
}}

DEMO_FIELDS = [
    ("Age","age","age",None),
    ("Gender","gender","choice",[(1,"पुरुष / Male"),(2,"महिला / Female"),(3,"अन्य / Other")]),
    ("Address","address","choice",[(1,"शहरी / Urban"),(2,"ग्रामीण / Rural")]),
    ("Education","edu","choice",[
        (1,"अशिक्षित / Illiterate"),
        (2,"प्राथमिक शिक्षा / Primary school certificate"),
        (3,"माध्यमिक शिक्षा / Middle school certificate"),
        (4,"हाईस्कूल / High school certificate"),
        (5,"इंटरमीडिएट या डिप्लोमा / Intermediate or Diploma"),
        (6,"स्नातक या स्नातकोत्तर / Graduate or Postgraduate"),
        (7,"व्यावसायिक डिग्री / Professional degree"),
    ]),
    ("Occupation","occ","choice",[
        (1,"छात्र-छात्रा / Student"),
        (2,"गृहिणी / Homemaker"),
        (3,"सरकारी कर्मचारी / Government employee"),
        (4,"निजी कर्मचारी / Private employee"),
        (5,"पेशेवर (डॉक्टर, इंजीनियर, शिक्षक आदि) / Professional (Doctor, Engineer, Teacher etc.)"),
        (6,"व्यवसाय / Business"),
        (7,"कुशल श्रमिक / Skilled worker"),
        (8,"अर्ध-कुशल श्रमिक / Semi-skilled worker"),
        (9,"अकुशल श्रमिक / Unskilled worker"),
        (10,"बेरोज़गार / Unemployed"),
        (11,"अन्य / Other"),
    ]),
    ("Income","income","choice",[
        (1,"₹77,000 या अधिक / ₹77,000 or more"),
        (2,"₹38,500 – ₹76,999"),
        (3,"₹28,900 – ₹38,499"),
        (4,"₹19,300 – ₹28,899"),
        (5,"₹7,700 – ₹19,299"),
        (6,"₹2,600 – ₹7,699"),
        (7,"₹2,599 या कम / ₹2,599 or less"),
    ]),
]

# ---------------------------------------------------------------------------
# Modified Kuppuswamy Socioeconomic Scale
# (per Dr. Ruchita Dixit's "Demographic and Socio-Economic Status" sheet)
#
# Education and Income categories above now match the official scale
# EXACTLY (word-for-word / bracket-for-bracket), so their scores below are
# exact, not approximated.
# Occupation still needs a best-effort mapping, because the demographic
# sheet's 11 occupation categories (Student, Homemaker, Govt/Private
# employee, Business, etc.) have no single official Kuppuswamy equivalent.
# Lines marked CONFIRM are worth double-checking with Dr. Dixit.
# ---------------------------------------------------------------------------
KUPPUSWAMY_EDUCATION_SCORE = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}

KUPPUSWAMY_OCCUPATION_SCORE = {
    1:1,   # Student -> Unemployed (1) -- CONFIRM with Dr. Dixit
    2:1,   # Homemaker -> Unemployed (1) -- CONFIRM with Dr. Dixit
    3:6,   # Government employee -> Semi-professional (6) -- CONFIRM (depends on post/grade)
    4:6,   # Private employee -> Semi-professional (6) -- CONFIRM (depends on position)
    5:10,  # Professional -> Profession (10)
    6:5,   # Business -> Clerical/Shop owner/Farmer (5) -- CONFIRM (depends on scale of business)
    7:4,   # Skilled worker -> Skilled worker (4)
    8:3,   # Semi-skilled worker -> Semi-skilled worker (3)
    9:2,   # Unskilled worker -> Unskilled worker (2)
    10:1,  # Unemployed -> Unemployed (1)
    11:2,  # Other -> Unskilled worker (2) -- CONFIRM
}

# Now exact, since the income brackets above are copied directly from the
# official scoring sheet.
KUPPUSWAMY_INCOME_SCORE = {1:12, 2:10, 3:6, 4:4, 5:3, 6:2, 7:1}

def kuppuswamy_class(total_score):
    if total_score >= 26: return "Upper Class (I) / उच्च वर्ग"
    if total_score >= 16: return "Upper Middle Class (II) / उच्च मध्यम वर्ग"
    if total_score >= 11: return "Lower Middle Class (III) / निम्न मध्यम वर्ग"
    if total_score >= 5:  return "Upper Lower Class (IV) / उच्च निम्न वर्ग"
    return "Lower Class (V) / निम्न वर्ग"

def calculate_kuppuswamy(education_code, occupation_code, income_code):
    edu_score = KUPPUSWAMY_EDUCATION_SCORE.get(education_code, 1)
    occ_score = KUPPUSWAMY_OCCUPATION_SCORE.get(occupation_code, 1)
    inc_score = KUPPUSWAMY_INCOME_SCORE.get(income_code, 1)
    total = edu_score + occ_score + inc_score
    return total, kuppuswamy_class(total)

# ---------------------------------------------------------------------------
# Comprehensive Laboratory Panel (CBC + Peripheral Smear)
# per Dr. Ruchita Dixit's "Lab_report.docx" cut-off table.
#
# The Iron Profile (Serum Ferritin, Serum Iron, TIBC, Transferrin, TSAT)
# has been removed entirely — it is no longer collected or used in any
# calculation. The IDA verdict is now derived purely from CBC + Peripheral
# Smear findings, per Dr. Dixit's instruction, since these are the tests
# realistically available to every patient (Iron Profile testing needs
# equipment/reagents many rural/resource-limited labs don't have).
# ---------------------------------------------------------------------------

# Numeric fields: (key, label, unit, default, min, max, step, cutoff_op, cutoff_value)
# cutoff_op/cutoff_value = None means informational only (no IDA cut-off given)
LAB_CBC_FIELDS = [
    ("hb",             "Hemoglobin (Hb)",       "g/dL",     12.0, 0.0, 25.0, 0.1, "<", 12.0),
    ("rbc_count",      "RBC Count",             "×10⁶/µL",  3.8,  0.0, 10.0, 0.1, "<", 3.8),
    ("hematocrit",     "Haematocrit (PCV)",     "%",        36.0, 0.0, 60.0, 0.5, "<", 36.0),
    ("mcv",            "MCV",                   "fL",       80.0, 0.0, 130.0,0.5, "<", 80.0),
    ("mch",            "MCH",                   "pg",       27.0, 0.0, 45.0, 0.5, "<", 27.0),
    ("mchc",           "MCHC",                  "g/dL",     32.0, 0.0, 40.0, 0.5, "<", 32.0),
    ("rdwcv",          "RDW-CV",                "%",        11.5, 0.0, 30.0, 0.1, ">", 14.5),
    ("wbc_count",      "Total WBC Count",       "/µL",      4000.0, 0.0, 30000.0, 100.0, None, None),
    ("platelet_count", "Platelet Count",        "lakh/µL",  1.5,  0.0, 10.0, 0.1, None, None),
]

# Categorical fields: (key, label, options, cutoff_value(s) suggestive of IDA)
LAB_PS_FIELDS = [
    ("rbc_size",            "RBC size",                   ["Microcytic","Normocytic","Macrocytic"], {"Microcytic"}),
    ("rbc_staining",        "RBC staining",               ["Hypochromic","Normochromic"], {"Hypochromic"}),
    ("anisocytosis",        "Anisocytosis",                ["Present","Absent"], {"Present"}),
    ("poikilocytosis",      "Poikilocytosis",              ["Present","Absent"], {"Present"}),
    ("pencil_cells",        "Pencil cells / Elliptocytes", ["Present","Absent"], {"Present"}),
    ("target_cells",        "Target cells",                 ["Present","Absent"], {"Absent"}),
    ("teardrop_cells",      "Teardrop cells",               ["Present","Absent"], {"Absent"}),
    ("schistocytes",        "Schistocytes",                 ["Present","Absent"], {"Absent"}),
    ("rouleaux",            "Rouleaux formation",           ["Present","Absent"], {"Absent"}),
    ("polychromasia",       "Polychromasia",                ["Present","Absent"], {"Absent"}),
    ("nucleated_rbc",       "Nucleated RBCs",               ["Present","Absent"], {"Absent"}),
    ("wbc_morphology",      "WBC morphology",               ["Normal","Abnormal"], {"Normal"}),
    ("platelet_morphology", "Platelet morphology",          ["Normal","Abnormal"], {"Normal"}),
    ("platelet_number",     "Platelet number",              ["Adequate","Increased","Decreased"], {"Adequate","Increased"}),
    ("hemoparasites",       "Hemoparasites",                ["Seen","Not seen"], {"Not seen"}),
]

# NOTE: the old 10-parameter "core count" system (CORE_CBC_PARAMS /
# CORE_PS_PARAMS / LAB_CORE_PARAMS) has been replaced by the simpler
# deterministic CBC-OR-PS rule in classify_lab_verdict() below — no more
# "X/5" or "X/10" fractional scoring.

def render_lab_panel_inputs(key_prefix, existing=None):
    """Renders the full CBC + Peripheral Smear input panel and returns a
    dict of entered values (key -> value). `existing` (optional dict)
    pre-fills values when updating an already-saved record."""
    existing = existing or {}

    st.markdown("**Complete Blood Count (CBC)**")
    values = {}
    ccols = st.columns(3)
    for i, (key, label, unit, default, lo, hi, step, op, cut) in enumerate(LAB_CBC_FIELDS):
        with ccols[i % 3]:
            start = existing.get(key, default)
            values[key] = st.number_input(f"{label} ({unit})", lo, hi, float(start), step, key=f"{key_prefix}_{key}")

    st.markdown("**Peripheral Smear (PS)**")
    st.caption("Note: at this hospital, Anisocytosis, Poikilocytosis & Pencil cells are only "
               "written in the report when PRESENT — if the report doesn't mention them at all, "
               "select **Absent** (don't leave as 'Not selected').")
    pcols = st.columns(3)
    for i, (key, label, options, cutset) in enumerate(LAB_PS_FIELDS):
        with pcols[i % 3]:
            opts_with_blank = ["Not selected"] + options
            start = existing.get(key, "Not selected")
            idx = opts_with_blank.index(start) if start in opts_with_blank else 0
            choice = st.selectbox(label, opts_with_blank, index=idx, key=f"{key_prefix}_{key}")
            values[key] = None if choice == "Not selected" else choice

    return values


def classify_lab_verdict(lab_values: dict, gender: int = None):
    """Simple, deterministic IDA lab verdict — CBC criterion OR PS criterion,
    either one being met is enough. No fractional "X/5" / "X/10" scoring —
    per Dr. Dixit's simplified rule.

    CBC criterion (gender-specific microcytic-hypochromic anaemia pattern):
        Female (gender==2): Hb < 12 g/dL AND MCV < 80 fL AND MCH < 27 pg
        Male   (gender==1): Hb < 13 g/dL AND MCV < 80 fL AND MCH < 27 pg
        (Gender==3/Other, or gender not provided: uses the female/lower
        12 g/dL cutoff as the safer, more inclusive default — confirm with
        Dr. Dixit if a different default is preferred.)

    PS criterion:
        RBC size == "Microcytic" AND RBC staining == "Hypochromic"
        (Per this hospital's reporting convention, Anisocytosis,
        Poikilocytosis and Pencil cells are only written in the report when
        PRESENT — so Microcytic + Hypochromic reliably implies the full
        classic IDA smear picture, without needing to check those 3
        separately.)

    Either CBC criterion OR PS criterion being met -> "Suggestive of IDA".
    This replaces the earlier 5/10-parameter core-count scoring system.
    Returns (verdict_string, detail_dict).
    """
    hb, mcv, mch = lab_values.get("hb"), lab_values.get("mcv"), lab_values.get("mch")
    rbc_size, rbc_staining = lab_values.get("rbc_size"), lab_values.get("rbc_staining")

    hb_cutoff = 13.0 if gender == 1 else 12.0  # Male=13, Female/Other/unknown=12 (safer default)

    cbc_entered = hb is not None and mcv is not None and mch is not None
    cbc_met = cbc_entered and (hb < hb_cutoff and mcv < 80.0 and mch < 27.0)

    ps_entered = rbc_size is not None and rbc_staining is not None
    ps_met = ps_entered and (rbc_size == "Microcytic" and rbc_staining == "Hypochromic")

    if not cbc_entered and not ps_entered:
        verdict = "Lab values not yet entered"
    elif cbc_met or ps_met:
        verdict = "Laboratory findings SUGGESTIVE of Iron Deficiency Anaemia"
    elif cbc_entered and ps_entered:
        verdict = "Laboratory findings NOT suggestive of Iron Deficiency Anaemia"
    else:
        missing = "Peripheral Smear (RBC size + staining)" if not ps_entered else "CBC (Hb, MCV, MCH)"
        verdict = f"Not suggestive so far — {missing} not yet entered; enter for a complete verdict"

    return verdict, {"cbc_met": cbc_met, "ps_met": ps_met, "cbc_entered": cbc_entered, "ps_entered": ps_entered}

def update_lab_values(record_id, lab_values, lab_confirmed):
    """Updates the full lab panel (CBC + Peripheral Smear)
    + Lab-Confirmed verdict for one existing record, identified by its
    record_id (used by 'Update Lab Report for an Existing Patient' — for
    iron profiles collected from outside labs and entered manually later)."""
    engine, _ = get_db_engine()
    init_db(engine)
    clean_lab = {k: _to_native(v) for k, v in lab_values.items() if v is not None and v != ""}
    with engine.begin() as conn:
        conn.execute(
            patients_table.update().where(patients_table.c.record_id == record_id).values(
                hemoglobin=clean_lab.get("hb"), mcv=clean_lab.get("mcv"),
                mch=clean_lab.get("mch"), mchc=clean_lab.get("mchc"),
                rdwcv=clean_lab.get("rdwcv"),
                lab_panel_json=json.dumps(clean_lab), lab_confirmed_ida=lab_confirmed,
            ))
        return True

SYMPTOM_FIELDS = [
    ("Paleness","असामान्य पीलापन / Paleness","choice",[
        (0,"त्वचा, हथेली या आँखों में पीलापन नहीं / No pallor seen"),
        (1,"हल्का पीलापन, केवल ध्यान देने पर दिखाई दे / Mild pallor noticed on examination"),
        (2,"स्पष्ट पीलापन, हथेली या आँखों में दिखने वाला / Obvious pallor of palms or conjunctiva"),
        (3,"पूरे शरीर में अधिक पीलापन / Marked generalized pallor")]),
    ("Fatigue","थकान / Fatigue","choice",[
        (0,"थकान नहीं / No fatigue"),
        (1,"अधिक काम करने के बाद थकान / Tired after heavy activity"),
        (2,"सामान्य दैनिक कार्यों में भी थकान / Fatigue during routine activities"),
        (3,"आराम की स्थिति में भी थकान / Fatigue even at rest")]),
    ("Tongue","जीभ में दर्द व सूजन / Sore Tongue","choice",[
        (0,"कोई दर्द या सूजन नहीं / No soreness or swelling"),
        (1,"हल्का दर्द या जलन / Mild soreness or burning sensation"),
        (2,"दर्द के कारण खाने में परेशानी / Pain affecting eating"),
        (3,"बहुत दर्द व सूजी हुई जीभ, खाने में कठिनाई / Severe pain or swollen tongue affecting food intake")]),
    ("Pica","मिट्टी, चॉक या बर्फ खाने की इच्छा / Pica","choice",[
        (0,"ऐसी इच्छा नहीं / No craving"),
        (1,"कभी-कभी इच्छा होना / Occasional craving"),
        (2,"बार-बार इच्छा होना / Frequent craving"),
        (3,"रोज़ाना गैर-खाद्य पदार्थ खाने की इच्छा या सेवन / Daily craving or consumption")]),
    ("EarNoise","कानों में घंटी या आवाज़ आना / Ringing in Ears","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी महसूस होना / Occasional episodes"),
        (2,"सप्ताह में कई बार / Several times per week"),
        (3,"लगातार या प्रतिदिन / Persistent or daily")]),
    ("BrittleNails","आसानी से टूटने वाले नाखून / Brittle Nails","choice",[
        (0,"सामान्य नाखून / Normal nails"),
        (1,"कभी-कभी टूटना / Occasional breakage"),
        (2,"बार-बार टूटना / Frequent breakage"),
        (3,"बहुत कमज़ोर नाखून या चम्मच जैसे नाखून / Severely brittle or spoon-shaped nails")]),
    ("Dizziness","चक्कर आना / Dizziness","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी, विशेषकर अचानक उठने पर / Occasional, especially on standing"),
        (2,"दैनिक गतिविधियों में चक्कर / During routine activities"),
        (3,"बार-बार चक्कर या बेहोशी जैसा लगना / Frequent dizziness or fainting sensation")]),
    ("RLS","पैरों में बेचैनी / Restless Legs","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी रात में / Occasional at night"),
        (2,"सप्ताह में कई रात / Several nights per week"),
        (3,"रोज़ रात को, नींद प्रभावित / Daily, affecting sleep")]),
    ("AngularStomatitis","मुँह के कोने फटना / Cracks at Corners of Mouth","choice",[
        (0,"नहीं / Absent"),
        (1,"हल्की दरारें / Small cracks"),
        (2,"दर्द वाली दरारें / Painful cracks"),
        (3,"संक्रमण के साथ गहरी दरारें / Deep cracks with infection")]),
    ("Dysphagia","निगलने में कठिनाई / Difficulty in Swallowing","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी परेशानी / Occasional difficulty"),
        (2,"बार-बार ठोस भोजन निगलने में परेशानी / Frequent difficulty with solids"),
        (3,"निगलने में गंभीर कठिनाई / Severe swallowing difficulty")]),
    ("Infections","बार-बार संक्रमण होना / Repeated Infections","choice",[
        (0,"सामान्य / No recurrent infections"),
        (1,"वर्ष में 1–2 बार / 1–2 episodes per year"),
        (2,"बार-बार संक्रमण (≥3 बार/वर्ष) / Recurrent infections (≥3/year)"),
        (3,"बार-बार गंभीर संक्रमण / Frequent severe infections")]),
    ("Bruising","हल्की चोट पर नीले निशान / Easy Bruising","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी चोट के बाद / Occasional after trauma"),
        (2,"छोटी चोटों पर बार-बार निशान / Frequent bruising after minor trauma"),
        (3,"बिना चोट के भी निशान / Spontaneous bruising")]),
    ("Irritability","चिड़चिड़ापन / Irritability","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी / Occasional"),
        (2,"बार-बार मूड परिवर्तन / Frequent irritability"),
        (3,"लगातार, दैनिक जीवन प्रभावित / Persistent, affecting daily life")]),
    ("Tachycardia","तेज़ धड़कन / Fast Heartbeat","choice",[
        (0,"सामान्य धड़कन / Normal heartbeat"),
        (1,"केवल अधिक मेहनत पर / Only after exertion"),
        (2,"हल्की गतिविधि पर भी / With mild activity"),
        (3,"आराम में भी तेज़ धड़कन / Even at rest")]),
    ("SOB","साँस फूलना / Shortness of Breath","choice",[
        (0,"नहीं / Absent"),
        (1,"अधिक मेहनत पर / On strenuous activity"),
        (2,"सामान्य चलने पर / During walking"),
        (3,"आराम में भी / At rest")]),
    ("Headache","सिरदर्द / Headache","choice",[
        (0,"नहीं / Absent"),
        (1,"कभी-कभी / Occasional"),
        (2,"बार-बार होने वाला / Frequent"),
        (3,"रोज़ाना या बहुत तेज़ / Daily or severe")]),
    ("PoorSleep","अच्छी नींद न आना / Poor Sleep","choice",[
        (0,"सामान्य नींद / Normal sleep"),
        (1,"कभी-कभी नींद खराब / Occasional disturbance"),
        (2,"सप्ताह में कई बार नींद प्रभावित / Several nights disturbed"),
        (3,"लगातार खराब नींद / Chronic poor sleep")]),
    ("ColdIntolerance","ठंड सहन न होना / Cold Intolerance","choice",[
        (0,"सामान्य सहनशीलता / Normal tolerance"),
        (1,"दूसरों से अधिक ठंड लगना / Feels colder than others"),
        (2,"अतिरिक्त कपड़ों की आवश्यकता / Needs extra clothing"),
        (3,"सामान्य तापमान में भी ठंड लगना / Cannot tolerate normal temperature")]),
    ("BlueSclera","आँखों का सफ़ेद भाग नीला दिखना / Blue Sclera","choice",[
        (0,"नहीं / Absent"),
        (1,"हल्का नीला रंग / Slight bluish tinge"),
        (2,"स्पष्ट नीला रंग / Clearly visible blue sclera"),
        (3,"बहुत अधिक नीला रंग / Marked blue sclera")]),
    ("HairLoss","बाल झड़ना / Hair Loss","choice",[
        (0,"सामान्य बाल / Normal hair"),
        (1,"बालों का हल्का झड़ना / Mild increased hair fall"),
        (2,"दिखाई देने वाला बाल पतला होना / Noticeable thinning"),
        (3,"बहुत अधिक बाल झड़ना / Marked hair loss")]),
    ("DrySkin","रूखी त्वचा / Dry Skin","choice",[
        (0,"सामान्य त्वचा / Normal skin"),
        (1,"हल्की रूखापन / Mild dryness"),
        (2,"रूखापन के साथ त्वचा पर पपड़ी बनना / Dryness with scaling"),
        (3,"फटी हुई या बहुत सूखी त्वचा / Severe cracked dry skin")]),
    ("Appetite","भूख कम लगना / Low Appetite","choice",[
        (0,"सामान्य भूख / Normal appetite"),
        (1,"थोड़ी कमी / Slight decrease"),
        (2,"सामान्य से कम खाना / Eats much less"),
        (3,"बहुत कम भूख या खाना मुश्किल / Marked loss of appetite")]),
]

if "lang" not in st.session_state: st.session_state.lang = "en"
if "view" not in st.session_state: st.session_state.view = "patient"
if "doctor_unlocked" not in st.session_state: st.session_state.doctor_unlocked = False
if "show_pw_box" not in st.session_state: st.session_state.show_pw_box = False
if "data_entry_unlocked" not in st.session_state: st.session_state.data_entry_unlocked = False
if "show_data_pw_box" not in st.session_state: st.session_state.show_data_pw_box = False
L = LANG[st.session_state.lang]

try:
    DOCTOR_PASSWORD = st.secrets["DOCTOR_PASSWORD"]
except Exception:
    DOCTOR_PASSWORD = "YTIdTlU6AMJyUp"  # local-only fallback; set DOCTOR_PASSWORD in secrets.toml / Streamlit Cloud Secrets to override

try:
    DATA_ENTRY_PASSWORD = st.secrets["DATA_ENTRY_PASSWORD"]
except Exception:
    DATA_ENTRY_PASSWORD = "gWdthmEiH71SUZ"  # local-only fallback; set DATA_ENTRY_PASSWORD in secrets.toml / Streamlit Cloud Secrets to override

# top controls
c1, c2, c3, c4 = st.columns([2, 1, 1, 1.3])
with c2:
    if st.button(L["lang_btn"], use_container_width=True):
        st.session_state.lang = "hi" if st.session_state.lang=="en" else "en"; st.rerun()
with c3:
    if st.session_state.view == "patient":
        if st.button(L["view_doc"], use_container_width=True):
            if st.session_state.doctor_unlocked:
                st.session_state.view = "doctor"
            else:
                st.session_state.show_pw_box = True
            st.rerun()
    elif st.session_state.view == "doctor":
        if st.button(L["view_pat"], use_container_width=True):
            st.session_state.view = "patient"; st.rerun()
with c4:
    if st.session_state.view != "data_entry":
        if st.button("📋 " + L["data_collection"], use_container_width=True):
            if st.session_state.data_entry_unlocked:
                st.session_state.view = "data_entry"
            else:
                st.session_state.show_data_pw_box = True
            st.rerun()
    else:
        if st.button(L["view_pat"], use_container_width=True):
            st.session_state.view = "patient"; st.rerun()

# Password prompt — Doctor View
if st.session_state.show_pw_box and not st.session_state.doctor_unlocked:
    with st.form("doctor_login", clear_on_submit=False):
        pw_col1, pw_col2 = st.columns([3,1])
        with pw_col1:
            entered_pw = st.text_input("🔒 Doctor password / डॉक्टर पासवर्ड", type="password", label_visibility="collapsed", placeholder="Enter doctor password")
        with pw_col2:
            pw_submit = st.form_submit_button("Unlock", use_container_width=True)
        if pw_submit:
            if entered_pw == DOCTOR_PASSWORD:
                st.session_state.doctor_unlocked = True
                st.session_state.view = "doctor"
                st.session_state.show_pw_box = False
                st.rerun()
            else:
                st.error("Incorrect password / गलत पासवर्ड")

# Password prompt — Data Collection
if st.session_state.show_data_pw_box and not st.session_state.data_entry_unlocked:
    with st.form("data_entry_login", clear_on_submit=False):
        dw_col1, dw_col2 = st.columns([3,1])
        with dw_col1:
            entered_dpw = st.text_input("🔒 Data collection password", type="password", label_visibility="collapsed", placeholder="Enter data collection password")
        with dw_col2:
            dw_submit = st.form_submit_button("Unlock", use_container_width=True)
        if dw_submit:
            if entered_dpw == DATA_ENTRY_PASSWORD:
                st.session_state.data_entry_unlocked = True
                st.session_state.view = "data_entry"
                st.session_state.show_data_pw_box = False
                st.rerun()
            else:
                st.error("Incorrect password / गलत पासवर्ड")

IS_DOCTOR = st.session_state.view == "doctor" and st.session_state.doctor_unlocked
IS_DATA_ENTRY = st.session_state.view == "data_entry" and st.session_state.data_entry_unlocked

if not IS_DATA_ENTRY:
    # brand bar — top-left, above the hero card (small logo + wordmark)
    st.markdown(f"""
    <div class="brandbar">{_logo_html}<span>IDA AI Screening</span></div>
    """, unsafe_allow_html=True)

    # hero
    st.markdown(f"""
    <div class="hero"><div class="hero-row">
      <div><h1>{L['title']}</h1><p>{L['subtitle']}</p></div>
    </div></div>
    """, unsafe_allow_html=True)

    # stats (doctor only)
    if IS_DOCTOR:
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-icon">🎯</div><div class="stat-label">{L['acc']}</div>
            <div class="stat-value">99.2%</div><div class="stat-sub">test records</div></div>
          <div class="stat-card"><div class="stat-icon">📈</div><div class="stat-label">{L['auc']}</div>
            <div class="stat-value">0.999</div><div class="stat-sub">near-perfect</div></div>
          <div class="stat-card"><div class="stat-icon">🗂️</div><div class="stat-label">{L['data']}</div>
            <div class="stat-value">20,000</div><div class="stat-sub">patients</div></div>
        </div>
        """, unsafe_allow_html=True)

    # form
    values = {}
    with st.form("ida_form"):
        st.markdown(f'<div class="section-card"><div class="section-title"><span class="icon">🧍</span>{L["demo"]}</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i,(col_name,key,kind,opts) in enumerate(DEMO_FIELDS):
            with cols[i%2]:
                label = L[key]
                if kind=="age":
                    values[col_name] = st.number_input(label, 1, 110, 30, 1)
                else:
                    lab = [f"{c} - {t}" for c,t in opts]
                    ch = st.selectbox(label, lab, key=f"d_{col_name}")
                    values[col_name] = int(ch.split(" - ")[0])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-card"><div class="section-title"><span class="icon">🩺</span>{L["symp"]}</div>', unsafe_allow_html=True)
        cols2 = st.columns(1)
        for i,(col_name,label,kind,opts) in enumerate(SYMPTOM_FIELDS):
            with cols2[0]:
                lab = [f"{c} - {t}" for c,t in opts]
                ch = st.selectbox(label, lab, key=f"s_{col_name}")
                values[col_name] = int(ch.split(" - ")[0])
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍  " + L["btn"], use_container_width=True, type="primary")

    # prediction
    if submitted:
        row = pd.DataFrame([[values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        if USES_SCALED and scaler:
            row = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLUMNS)
        pred  = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        conf  = proba*100 if pred==1 else (1-proba)*100
        cls   = "present" if pred==1 else "absent"
        icon  = "⚠️" if pred==1 else "✅"
        title = L["present"] if pred==1 else L["absent"]
        advice= L["advice_p"] if pred==1 else L["advice_a"]

        st.markdown(f"""
        <div class="result-box {cls}">
          <div class="result-icon-wrap">{icon}</div>
          <div style="flex:1">
            <p class="result-title">{title}</p>
            <p class="result-advice">{advice}</p>
            <div class="gauge-wrap">
              <div class="gauge-track"><div class="gauge-fill {cls}" style="width:{conf:.0f}%"></div></div>
              <div class="gauge-label">{L['conf']}: {conf:.1f}%</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if IS_DOCTOR:
            kup_score, kup_class = calculate_kuppuswamy(values["Education"], values["Occupation"], values["Income"])
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px; padding:14px 18px; margin-top:12px;">
              <div style="font-size:11px; color:#9a9aae; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px;">
                📋 Socioeconomic Status — Modified Kuppuswamy Scale
              </div>
              <div style="font-size:16px; font-weight:700; color:#e8e8f0;">{kup_class}</div>
              <div style="font-size:12px; color:#8a8aa0; margin-top:2px;">Total score: {kup_score} / 29</div>
            </div>
            """, unsafe_allow_html=True)


            try:
                with st.expander("🔎 " + L["why"], expanded=True):
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(row)
                    sv_p = sv[1][0] if isinstance(sv,list) else (sv[0,:,1] if sv.ndim==3 else sv[0])
                    contrib = pd.Series(sv_p, index=FEATURE_COLUMNS)
                    contrib = contrib[contrib != 0].sort_values()
                    if len(contrib):
                        colors = ["#e74c3c" if v>0 else "#2ecc71" for v in contrib.values]
                        fig, ax = plt.subplots(figsize=(7, max(3,len(contrib)*0.32)))
                        fig.patch.set_facecolor("#15151f"); ax.set_facecolor("#15151f")
                        contrib.plot(kind="barh", ax=ax, color=colors)
                        ax.axvline(0, color="#555", lw=0.8, ls="--")
                        ax.tick_params(colors="#ccc", labelsize=9)
                        ax.set_xlabel("Impact", color="#ccc")
                        ax.spines[:].set_color("#444")
                        plt.tight_layout()
                        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig); buf.seek(0)
                        st.image(buf, use_container_width=True)
                    st.caption(L["why_cap"])
            except Exception as e:
                st.info(f"Chart unavailable: {e}")

    def img_bytes(p):
        with open(p,"rb") as f: return f.read()

    if IS_DOCTOR:
        st.markdown(f'<div class="section-card"><div class="section-title"><span class="icon">📊</span>Model analytics</div>', unsafe_allow_html=True)
        if os.path.exists("feature_importance.png"):
            with st.expander(L["fi"]): st.image(img_bytes("feature_importance.png"), use_container_width=True)
        if os.path.exists("shap_summary.png"):
            with st.expander(L["shap"]): st.image(img_bytes("shap_summary.png"), use_container_width=True)
        if os.path.exists("model_comparison.png"):
            with st.expander(L["cmp"]): st.image(img_bytes("model_comparison.png"), use_container_width=True)
        if os.path.exists("roc_curve.png"):
            with st.expander("📉 ROC Curve"): st.image(img_bytes("roc_curve.png"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer-box">{L["disclaimer"]}</div>', unsafe_allow_html=True)

else:
    # =========================================================================
    # DATA COLLECTION PAGE (password-protected, separate from Doctor View)
    # For enrolling real patients: demographics + symptoms + lab-confirmed
    # IDA status, stored in a local SQLite database for later analysis.
    # =========================================================================
    st.markdown(f"""
    <div class="brandbar">{_logo_html}<span>IDA AI Screening</span></div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero"><div class="hero-row">
      <div class="hero-icon">📋</div>
      <div><h1>Patient Data Collection</h1><p>Enroll a patient with demographics, symptoms, and lab-confirmed diagnosis</p></div>
    </div></div>
    """, unsafe_allow_html=True)

    _engine, _is_cloud = get_db_engine()
    if _is_cloud:
        st.success("☁️ Connected to cloud database — patient records are stored permanently and will survive app restarts.")
    else:
        st.warning("⚠️ No cloud database configured — using local storage only. This data is **not guaranteed to persist** if deployed on Streamlit Cloud. See README to connect a permanent database. Export records regularly using the button below as a backup.")
        with st.expander("Technical details / retry connection"):
            st.code(st.session_state.get("db_error", "No SUPABASE_DB_URL found in Secrets, or it could not be read."))
            if st.button("🔄 Retry cloud connection"):
                _try_cloud_engine.clear()
                st.rerun()

    tab1, tab2 = st.tabs(["➕ Add New Patient Record", "📄 View / Export Records"])

    with tab1:
        de_values = {}
        with st.form("data_entry_form"):
            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🆔</span>Patient Identification</div>', unsafe_allow_html=True)
            idcol1, idcol2, idcol3 = st.columns(3)
            with idcol1:
                patient_id = st.text_input("Patient ID / Registration Number")
            with idcol2:
                patient_name = st.text_input("Patient Name (optional)")
            with idcol3:
                patient_phone = st.text_input("Phone Number (optional)")
            consent_given = st.checkbox(
                "✅ Patient / guardian has given verbal or written consent to record and use this "
                "health information for the IDA screening study.",
                key="de_consent")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🧍</span>Demographics</div>', unsafe_allow_html=True)
            dcols = st.columns(2)
            for i,(col_name,key,kind,opts) in enumerate(DEMO_FIELDS):
                with dcols[i%2]:
                    label = L[key]
                    if kind=="age":
                        de_values[col_name] = st.number_input(label, 1, 110, 30, 1, key=f"de_{col_name}")
                    else:
                        lab = [f"{c} - {t}" for c,t in opts]
                        ch = st.selectbox(label, lab, key=f"de_d_{col_name}")
                        de_values[col_name] = int(ch.split(" - ")[0])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🩺</span>Signs & Symptoms</div>', unsafe_allow_html=True)
            scols = st.columns(1)
            for i,(col_name,label,kind,opts) in enumerate(SYMPTOM_FIELDS):
                with scols[0]:
                    lab = [f"{c} - {t}" for c,t in opts]
                    ch = st.selectbox(label, lab, key=f"de_s_{col_name}")
                    de_values[col_name] = int(ch.split(" - ")[0])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🧪</span>Laboratory Report — CBC & Peripheral Smear</div>', unsafe_allow_html=True)
            lab_available = st.checkbox("Lab report available now? / लैब रिपोर्ट अभी उपलब्ध है?", value=False,
                help="Leave unchecked if the patient has just been registered and sent for testing — you can come back and add the values later via 'Update Lab Report for an Existing Patient'.")
            lab_vals = render_lab_panel_inputs("de")
            st.caption("The laboratory verdict (Suggestive / Not suggestive of IDA) is calculated "
                       "automatically from these values against Dr. Dixit's cut-off table — no manual selection needed.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">👨‍⚕️</span>Clinician\'s Assessment (Human Judgement)</div>', unsafe_allow_html=True)
            st.caption("This system is a screening aid, not a replacement for clinical judgement. "
                       "If you (the clinician) have your own impression of this patient based on "
                       "examination — independent of the AI prediction — record it here.")
            clinician_assessment = st.selectbox(
                "Clinician's own assessment",
                ["Not assessed", "IDA Positive (Clinical Judgement + Lab Reports)", "IDA Negative (Clinical Judgement + Lab Reports)", "Uncertain / Needs further workup"],
                key="clinician_assessment_input")
            clinician_notes = st.text_area("Clinician's notes (optional)", key="clinician_notes_input", height=70)
            st.markdown('</div>', unsafe_allow_html=True)

            de_submitted = st.form_submit_button("💾 Save Patient Record", use_container_width=True, type="primary")

        def _do_save(pid_clean, kup_score, kup_class):
            de_row = pd.DataFrame([[de_values[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            de_row_final = pd.DataFrame(scaler.transform(de_row), columns=FEATURE_COLUMNS) if (USES_SCALED and scaler) else de_row
            ai_pred = model.predict(de_row_final)[0]
            ai_proba = model.predict_proba(de_row_final)[0][1]
            ai_pred_label = "IDA Positive" if ai_pred == 1 else "IDA Negative"

            if lab_available:
                lab_confirmed, lab_detail = classify_lab_verdict(lab_vals, gender=de_values.get("Gender"))
                lab_vals_to_save = lab_vals
            else:
                lab_confirmed = "Pending — awaiting lab report"
                lab_vals_to_save = {}  # don't store placeholder defaults if lab wasn't actually done

            save_patient_record(pid_clean, patient_name.strip(), de_values, lab_vals_to_save,
                                 lab_confirmed, ai_pred_label, round(ai_proba*100, 1),
                                 kup_score, kup_class, patient_phone.strip() if patient_phone else None,
                                 clinician_assessment, clinician_notes.strip() if clinician_notes else None,
                                 consent_given)
            st.session_state["dup_pending"] = None
            st.success(f"✅ Record saved for Patient ID: {pid_clean}")
            st.info(f"Lab verdict: **{lab_confirmed}** — Clinician's assessment: **{clinician_assessment}**")
            # Note: the AI prediction (ai_pred_label / ai_proba) is still computed
            # and saved to the database above for later research use, but is not
            # shown here for now — per Dr. Dixit's instruction, since the model is
            # trained on synthetic data only and hasn't been validated on real
            # patients yet.

        # Socioeconomic (Kuppuswamy) classification — calculated once here (not
        # inside _do_save) so it's available both for saving AND for display in
        # the duplicate-ID confirmation box below, even before "Save Anyway" is
        # clicked. (Previously computed only inside _do_save, which crashed the
        # page with a NameError the moment a duplicate Patient ID was entered —
        # a very real scenario for follow-up visits — since the duplicate-warning
        # box referenced kup_score/kup_class before _do_save ever ran.)
        kup_score, kup_class = calculate_kuppuswamy(de_values["Education"], de_values["Occupation"], de_values["Income"])

        if de_submitted:
            if not patient_id.strip():
                st.error("Patient ID is required.")
            elif not consent_given:
                st.error("⚠️ Please confirm patient/guardian consent before saving this record.")
            else:
                pid_clean = patient_id.strip()
                dup_records = patient_id_exists(pid_clean)
                if dup_records:
                    st.session_state["dup_pending"] = {
                        "pid": pid_clean, "count": len(dup_records), "last_ts": dup_records[-1].entry_timestamp}
                else:
                    _do_save(pid_clean, kup_score, kup_class)

        dup = st.session_state.get("dup_pending")
        if dup:
            st.error(f"⚠️ Patient ID **'{dup['pid']}'** already has {dup['count']} record(s) saved "
                      f"(most recent: {dup['last_ts']}). If this is a follow-up visit for the same "
                      "patient, use 'Update Lab Report for an Existing Patient' in the Records tab "
                      "instead of creating a new record.")
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.checkbox("This is genuinely a different patient who happens to share this ID",
                            key="dup_override_checkbox")
            with col_b:
                if st.button("💾 Save Anyway", use_container_width=True, disabled=not st.session_state.get("dup_override_checkbox")):
                    _do_save(dup["pid"], kup_score, kup_class)
                    st.rerun()
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px; padding:14px 18px; margin-top:6px;">
                  <div style="font-size:11px; color:#9a9aae; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px;">
                    📋 Socioeconomic Status — Modified Kuppuswamy Scale (auto-calculated)
                  </div>
                  <div style="font-size:16px; font-weight:700; color:#e8e8f0;">{kup_class}</div>
                  <div style="font-size:12px; color:#8a8aa0; margin-top:2px;">Total score: {kup_score} / 29</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        records_df = load_all_records(active_only=True)
        if len(records_df) == 0:
            st.info("No active patient records. (Check below in case all records are hidden.)")
        else:
            st.write(f"**Total records: {len(records_df)}**")

            # --- Build the full, analysis-ready export table -------------------------
            # Flatten the lab_panel_json blob into individual readable columns, and
            # put Lab values + AI prediction LAST in the column order (per Dr. Dixit).
            lab_key_order = [k for k,*_ in LAB_CBC_FIELDS] + [k for k,*_ in LAB_PS_FIELDS]
            lab_label = {k: lab for k, lab, *_ in LAB_CBC_FIELDS}
            lab_label.update({k: lab for k, lab, *_ in LAB_PS_FIELDS})

            def flatten_lab(js):
                try:
                    d = json.loads(js) if js else {}
                except Exception:
                    d = {}
                return {f"Lab: {lab_label[k]}": d.get(k) for k in lab_key_order}

            lab_cols_df = records_df["lab_panel_json"].apply(flatten_lab).apply(pd.Series) if "lab_panel_json" in records_df.columns else pd.DataFrame(index=records_df.index)

            id_cols   = ["record_id","patient_id","patient_name","patient_phone","entry_timestamp"]
            demo_cols = [c for c,*_ in DEMO_FIELDS]
            symp_cols = [c for c,*_ in SYMPTOM_FIELDS]
            other_cols = ["kuppuswamy_score","kuppuswamy_class","clinician_assessment","clinician_notes"]
            # Lab + AI prediction go LAST, as requested
            last_cols = (["hemoglobin","mcv","mch","mchc","rdwcv"]
                         + list(lab_cols_df.columns) + ["lab_confirmed_ida","ai_predicted_ida","ai_confidence"])

            export_df = pd.concat([records_df, lab_cols_df], axis=1)
            ordered = [c for c in (id_cols+demo_cols+symp_cols+other_cols+last_cols) if c in export_df.columns]
            export_df = export_df[ordered]

            # --- Compact overview table (few columns -> no horizontal scroll) --------
            # AI prediction intentionally excluded from this live view for now (per
            # Dr. Dixit's instruction) — the model is trained on synthetic data only
            # and hasn't been validated on real patients yet, so its live +/- output
            # isn't shown to avoid confusing interns/clinicians during data
            # collection. It is still computed and saved in the database/Excel
            # export for later research use once real-patient validation is done.
            overview_cols = ["record_id","patient_id","patient_name","patient_phone",
                              "entry_timestamp","clinician_assessment","lab_confirmed_ida"]
            overview_cols = [c for c in overview_cols if c in records_df.columns]
            st.dataframe(records_df[overview_cols], use_container_width=True, hide_index=True)

            # --- Export: Excel (recommended — Hindi text stays intact, easy to
            # analyse/filter) with CSV offered as a fallback -----------------------
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Patient Records")
            ex1, ex2 = st.columns(2)
            with ex1:
                st.download_button("⬇️ Download as Excel (recommended)", xls_buf.getvalue(),
                                    "patient_records_export.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)
            with ex2:
                csv_data = export_df.to_csv(index=False).encode("utf-8-sig")  # BOM so Hindi text opens correctly if double-clicked in Excel
                st.download_button("⬇️ Download as CSV", csv_data, "patient_records_export.csv",
                                    "text/csv", use_container_width=True)
            st.caption("Excel is recommended for analysis in this study — it keeps the Hindi text and "
                       "column formatting intact. Lab values and AI/clinical verdicts are placed as the "
                       "last columns in both files.")

            # --- Per-patient vertical detail view (easier to read than one wide table) --
            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🔍</span>View Full Record (one patient)</div>', unsafe_allow_html=True)
            detail_options = [f"{r.record_id} — {r.patient_id} — {r.patient_name or '(no name)'}" for r in records_df.itertuples()]
            detail_selected = st.selectbox("Select a patient to view all details", detail_options, key="detail_view_select")
            detail_id = int(detail_selected.split(" — ")[0])
            detail_row = export_df[export_df["record_id"] == detail_id].iloc[0]

            def show_section(title, cols, show_empty=False):
                if show_empty:
                    present = [c for c in cols if c in detail_row.index]
                else:
                    present = [c for c in cols if c in detail_row.index and pd.notna(detail_row[c]) and detail_row[c] != ""]
                if not present:
                    return
                st.markdown(f"**{title}**")
                for c in present:
                    disp_label = c.replace("Lab: ", "") if c.startswith("Lab: ") else c
                    val = detail_row[c]
                    val_display = "<span style='color:#a8a8ba;font-weight:600;'>Not entered</span>" if (pd.isna(val) or val == "") else val
                    st.markdown(f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>"
                                f"<span style='color:#b4b4c8;font-weight:500;'>{disp_label}</span><span style='color:#f0f0f6;font-weight:600;'>{val_display}</span></div>",
                                unsafe_allow_html=True)

            show_section("🆔 Identification", id_cols)
            show_section("🧍 Demographics", demo_cols)
            show_section("🩺 Signs & Symptoms", symp_cols)
            show_section("📋 Socioeconomic / Clinician", other_cols)
            # show_empty=True so missing CBC/PS fields show as "Not entered" instead
            # of silently disappearing — this makes gaps in the lab panel (e.g. PS
            # fields not filled in) visible to whoever reviews the record, instead
            # of looking like the field was never part of the form at all.
            # AI prediction intentionally excluded here — see note below.
            lab_display_cols = ["hemoglobin","mcv","mch","mchc","rdwcv"] + list(lab_cols_df.columns) + ["lab_confirmed_ida"]
            show_section("🧪 Laboratory Report (CBC + Peripheral Smear) & Verdict", lab_display_cols, show_empty=True)
            st.markdown('</div>', unsafe_allow_html=True)

            options = [f"{r.record_id} — {r.patient_id} — {r.patient_name or '(no name)'}" for r in records_df.itertuples()]

            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🧪</span>Update Lab Report for an Existing Patient</div>', unsafe_allow_html=True)
            st.caption("Use this once the patient's CBC / Peripheral Smear report comes back "
                       "from the outside lab. Select their record below, enter the values, and the "
                       "laboratory verdict will be calculated automatically.")
            lab_selected = st.selectbox("Select patient to update", options, key="lab_update_select")
            lab_selected_id = int(lab_selected.split(" — ")[0])
            lab_selected_row = records_df[records_df["record_id"] == lab_selected_id].iloc[0]
            try:
                existing_lab = json.loads(lab_selected_row.get("lab_panel_json") or "{}")
            except Exception:
                existing_lab = {}

            with st.form("update_lab_form"):
                up_lab_vals = render_lab_panel_inputs("up", existing=existing_lab)
                update_submitted = st.form_submit_button("💾 Save Lab Report for this Patient", use_container_width=True, type="primary")

            if update_submitted:
                lab_confirmed, lab_detail = classify_lab_verdict(up_lab_vals, gender=lab_selected_row.get("Gender"))
                update_lab_values(lab_selected_id, up_lab_vals, lab_confirmed)
                st.success(f"Lab report saved for record {lab_selected_id}. Result: **{lab_confirmed}**")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card"><div class="section-title"><span class="icon">🗂️</span>Manage a Record</div>', unsafe_allow_html=True)
            selected = st.selectbox("Select a record to hide or delete", options, key="manage_select")
            selected_id = int(selected.split(" — ")[0])

            mcol1, mcol2 = st.columns(2)
            with mcol1:
                if st.button("🙈 Hide this record", use_container_width=True,
                              help="Removes it from this list, but keeps it safely in the database. Can be restored anytime."):
                    hide_record(selected_id)
                    st.success(f"Record {selected_id} hidden.")
                    st.rerun()
            with mcol2:
                confirm_delete = st.checkbox("I understand this cannot be undone", key="confirm_del")
                if st.button("🗑️ Permanently Delete", use_container_width=True, disabled=not confirm_delete):
                    delete_record_permanently(selected_id)
                    st.success(f"Record {selected_id} permanently deleted.")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        hidden_df = load_all_records(active_only=False)
        if len(hidden_df) > 0:
            with st.expander(f"🙈 Hidden records ({len(hidden_df)}) — click to view and restore"):
                st.dataframe(hidden_df, use_container_width=True)
                hidden_options = [f"{r.record_id} — {r.patient_id} — {r.patient_name or '(no name)'}" for r in hidden_df.itertuples()]
                restore_selected = st.selectbox("Select a record to restore", hidden_options, key="restore_select")
                restore_id = int(restore_selected.split(" — ")[0])
                if st.button("♻️ Restore this record", key="restore_btn"):
                    restore_record(restore_id)
                    st.success(f"Record {restore_id} restored.")
                    st.rerun()
