"""
gui_app.py  —  IDA AI Screening Tool (Upgraded Desktop GUI)
============================================================
Upgraded per faculty feedback:
  1. Crisp outcome: "Iron Deficiency: Present / Absent"
  2. Hindi / English language toggle
  3. Patient View (simple result only) vs Doctor View (full charts + analysis)
  4. Dashboard-style UI with medical theme

Run train_model.py FIRST, then:  python gui_app.py
"""

import os, sys, io, json, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
for path in ["ida_model.joblib", "feature_list.joblib"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Please run 'python train_model.py' first.")
        sys.exit(1)

model          = joblib.load("ida_model.joblib")
FEATURE_COLUMNS = joblib.load("feature_list.joblib")
scaler         = joblib.load("scaler.joblib")      if os.path.exists("scaler.joblib")     else None
model_meta     = joblib.load("model_meta.joblib")  if os.path.exists("model_meta.joblib") else {"name":"Model","uses_scaled_input":False}
USES_SCALED    = model_meta.get("uses_scaled_input", False)

# ---------------------------------------------------------------------------
# Doctor View password (local desktop app — change this before sharing the
# .exe / source with others if you want a different password. This must
# be kept in sync with DOCTOR_PASSWORD in web_app.py / secrets.toml if you
# want the same password on both apps.)
# ---------------------------------------------------------------------------
DOCTOR_PASSWORD = "YTIdTlU6AMJyUp"

# ---------------------------------------------------------------------------
# Patient Data Collection — Cloud database (Supabase/Postgres), with a
# local SQLite fallback. Both apps (web_app.py and gui_app.py) use the
# SAME cloud database if configured, so records are shared between them.
#
# To connect this desktop app to the cloud database: create a file named
# "db_config.txt" in this same folder (NOT uploaded to GitHub — see
# README) containing just the connection URL on one line, e.g.:
#   postgresql://postgres:yourpassword@your-project.supabase.co:5432/postgres
# If that file doesn't exist, it automatically falls back to a local
# SQLite file (patient_records.db) — no setup needed for local testing.
# ---------------------------------------------------------------------------
DATA_ENTRY_PASSWORD = "gWdthmEiH71SUZ"

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float

def get_db_engine():
    config_path = "db_config.txt"
    if os.path.exists(config_path):
        with open(config_path) as f:
            db_url = f.read().strip()
        if db_url:
            try:
                engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
                with engine.connect():
                    pass
                return engine, True
            except Exception:
                pass  # fall through to local SQLite below
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

def init_db(engine):
    _metadata.create_all(engine, checkfirst=True)
    ensure_is_active_column(engine)
    ensure_kuppuswamy_columns(engine)
    ensure_clinician_columns(engine)
    ensure_lab_panel_column(engine)
    ensure_consent_column(engine)
    ensure_cbc_core_columns(engine)

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

def ensure_lab_panel_column(engine):
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    cols = [c["name"] for c in inspector.get_columns("patients")]
    if "lab_panel_json" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE patients ADD COLUMN lab_panel_json VARCHAR"))

def _to_native(v):
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
    """Returns existing active records with this exact Patient ID (used
    to warn staff before creating an accidental duplicate)."""
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
        query = patients_table.select().order_by(patients_table.c.record_id.desc()).where(
            patients_table.c.is_active == (1 if active_only else 0)
        ).with_only_columns(
            patients_table.c.record_id, patients_table.c.patient_id, patients_table.c.patient_name,
            patients_table.c.entry_timestamp, patients_table.c.lab_confirmed_ida
            # ai_predicted_ida intentionally excluded from this list view for now
            # (per Dr. Dixit's instruction) — still saved in the DB/export, just
            # not shown live during data collection. See recalculate_lab_verdicts.py
            # notes / README for context.
        )
        df = pd.read_sql_query(query, conn)
    return [tuple(r) for r in df.itertuples(index=False, name=None)]

def get_full_record(record_id):
    engine, _ = get_db_engine()
    init_db(engine)
    with engine.connect() as conn:
        sel = patients_table.select().where(patients_table.c.record_id == record_id)
        row = conn.execute(sel).fetchone()
        return row._mapping if row is not None else None

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

def _build_export_dataframe():
    """Builds the analysis-ready export table: flattens lab_panel_json into
    readable columns and puts Lab values + AI prediction LAST, mirroring
    web_app.py's export."""
    engine, _ = get_db_engine()
    init_db(engine)
    with engine.connect() as conn:
        records_df = pd.read_sql_query(patients_table.select(), conn)
    if len(records_df) == 0:
        return records_df

    lab_key_order = [k for k, *_ in LAB_CBC_FIELDS] + [k for k, *_ in LAB_PS_FIELDS]
    lab_label = {k: lab for k, lab, *_ in LAB_CBC_FIELDS}
    lab_label.update({k: lab for k, lab, *_ in LAB_PS_FIELDS})

    def flatten_lab(js):
        try:
            d = json.loads(js) if js else {}
        except Exception:
            d = {}
        return {f"Lab: {lab_label[k]}": d.get(k) for k in lab_key_order}

    lab_cols_df = records_df["lab_panel_json"].apply(flatten_lab).apply(pd.Series) if "lab_panel_json" in records_df.columns else pd.DataFrame(index=records_df.index)
    id_cols = ["record_id", "patient_id", "patient_name", "patient_phone", "entry_timestamp"]
    demo_cols = [c for c, *_ in DEMO_FIELDS]
    symp_cols = [c for c, *_ in SYMPTOM_FIELDS]
    other_cols = ["kuppuswamy_score", "kuppuswamy_class", "clinician_assessment", "clinician_notes"]
    last_cols = (["hemoglobin", "mcv", "mch", "mchc", "rdwcv"]
                 + list(lab_cols_df.columns) + ["lab_confirmed_ida", "ai_predicted_ida", "ai_confidence"])
    export_df = pd.concat([records_df, lab_cols_df], axis=1)
    ordered = [c for c in (id_cols + demo_cols + symp_cols + other_cols + last_cols) if c in export_df.columns]
    return export_df[ordered]

def export_all_records_csv(path):
    df = _build_export_dataframe()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)

def export_all_records_excel(path):
    df = _build_export_dataframe()
    df.to_excel(path, index=False, sheet_name="Patient Records")
    return len(df)

# ---------------------------------------------------------------------------
# Language strings
# ---------------------------------------------------------------------------
LANG = {
    "en": {
        "title":        "IDA Screening — AI Diagnostic Tool",
        "subtitle":     "Signs & symptoms only · No blood test required",
        "lang_btn":     "🇮🇳 हिंदी",
        "view_btn_doc": "🩺 Doctor View",
        "view_btn_pat": "👤 Patient View",
        "demo_head":    "PATIENT DEMOGRAPHICS",
        "symp_head":    "SIGNS & SYMPTOMS",
        "predict_btn":  "🔍 Run AI Prediction",
        "present":      "Iron Deficiency: Present",
        "absent":       "Iron Deficiency: Absent",
        "conf":         "Confidence",
        "advice_p":     "⚠ Please consult a doctor and confirm with a blood test.",
        "advice_a":     "✓ No iron deficiency indicated. See a doctor if symptoms persist.",
        "disclaimer":   "Screening aid only — not a substitute for clinical diagnosis.",
        "chart_lbl":    "View Chart:",
        "chart_fi":     "Feature Importance",
        "chart_shap":   "SHAP Summary",
        "chart_cmp":    "Model Comparison",
        "chart_pat":    "This Patient",
        "view_chart":   "View",
        "age": "Age (years)", "gender": "Gender", "address": "Address",
        "edu": "Education",   "occ": "Occupation", "income": "Monthly Income (₹)",
    },
    "hi": {
        "title":        "IDA जांच — AI निदान उपकरण",
        "subtitle":     "केवल लक्षणों पर · रक्त परीक्षण नहीं",
        "lang_btn":     "🇬🇧 English",
        "view_btn_doc": "🩺 डॉक्टर दृश्य",
        "view_btn_pat": "👤 मरीज़ दृश्य",
        "demo_head":    "मरीज़ की जानकारी",
        "symp_head":    "संकेत और लक्षण",
        "predict_btn":  "🔍 AI जांच करें",
        "present":      "आयरन की कमी: उपस्थित",
        "absent":       "आयरन की कमी: अनुपस्थित",
        "conf":         "विश्वास",
        "advice_p":     "⚠ कृपया डॉक्टर से मिलें और रक्त परीक्षण से पुष्टि करें।",
        "advice_a":     "✓ आयरन की कमी नहीं दिखती। लक्षण रहें तो डॉक्टर से मिलें।",
        "disclaimer":   "यह केवल जांच सहायक है — चिकित्सीय निदान का विकल्प नहीं।",
        "chart_lbl":    "चार्ट देखें:",
        "chart_fi":     "फीचर महत्व",
        "chart_shap":   "SHAP सारांश",
        "chart_cmp":    "मॉडल तुलना",
        "chart_pat":    "यह मरीज़",
        "view_chart":   "देखें",
        "age": "आयु (वर्ष)", "gender": "लिंग", "address": "पता",
        "edu": "शिक्षा",     "occ": "व्यवसाय", "income": "मासिक आय (₹)",
    }
}

# ---------------------------------------------------------------------------
# Field definitions (bilingual)
# ---------------------------------------------------------------------------
DEMO_FIELDS = [
    ("Age",       "age",    "age",    None),
    ("Gender",    "gender", "choice", [(1,"पुरुष / Male"),(2,"महिला / Female"),(3,"अन्य / Other")]),
    ("Address",   "address","choice", [(1,"शहरी / Urban"),(2,"ग्रामीण / Rural")]),
    ("Education", "edu",    "choice", [
        (1,"अशिक्षित / Illiterate"),
        (2,"प्राथमिक शिक्षा / Primary school certificate"),
        (3,"माध्यमिक शिक्षा / Middle school certificate"),
        (4,"हाईस्कूल / High school certificate"),
        (5,"इंटरमीडिएट या डिप्लोमा / Intermediate or Diploma"),
        (6,"स्नातक या स्नातकोत्तर / Graduate or Postgraduate"),
        (7,"व्यावसायिक डिग्री / Professional degree"),
    ]),
    ("Occupation","occ",    "choice", [
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
    ("Income",    "income", "choice", [
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
# Modified Kuppuswamy Socioeconomic Scale (same logic as web_app.py)
# Education and Income now map EXACTLY to Dr. Dixit's scoring sheet.
# Occupation still needs a best-effort mapping (lines marked CONFIRM) since
# the 11 demographic-sheet categories have no single official equivalent.
# ---------------------------------------------------------------------------
KUPPUSWAMY_EDUCATION_SCORE = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}
KUPPUSWAMY_OCCUPATION_SCORE = {
    1:1, 2:1,      # Student, Homemaker -> Unemployed (1) -- CONFIRM
    3:6, 4:6,      # Government / Private employee -> Semi-professional (6) -- CONFIRM
    5:10,          # Professional -> Profession (10)
    6:5,           # Business -> Clerical/Shop owner/Farmer (5) -- CONFIRM
    7:4, 8:3, 9:2, 10:1,  # Skilled, Semi-skilled, Unskilled, Unemployed
    11:2,          # Other -> Unskilled (2) -- CONFIRM
}
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

def update_lab_values(record_id, lab_values, lab_confirmed):
    """Updates the full lab panel (CBC + Peripheral Smear)
    + Lab-Confirmed verdict for one existing record (for reports collected
    from outside labs, entered manually later once the patient has already
    been enrolled)."""
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

# ---------------------------------------------------------------------------
# Comprehensive Laboratory Panel (CBC + Peripheral Smear)
# per Dr. Ruchita Dixit's "Lab_report.docx" cut-off table. Kept identical
# to web_app.py so both interfaces produce the same verdict.
#
# The Iron Profile (Serum Ferritin, Serum Iron, TIBC, Transferrin, TSAT)
# has been removed entirely — it is no longer collected or used in any
# calculation. The IDA verdict is now derived purely from CBC + Peripheral
# Smear findings, per Dr. Dixit's instruction.
# ---------------------------------------------------------------------------
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

def classify_lab_verdict(lab_values: dict, gender: int = None):
    """Simple, deterministic IDA lab verdict — CBC criterion OR PS criterion,
    either one being met is enough. No fractional "X/5" / "X/10" scoring.
    Identical logic to web_app.py's classify_lab_verdict — see there for
    the full doc-comment.
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

def collect_lab_panel_values(widgets):
    """Reads all lab_* Tkinter variables (populated by the CBC/PS
    section) into a single dict of key -> value, ready for classify_lab_verdict()."""
    values = {}
    for key, label, unit, default, lo, hi, step, op, cut in LAB_CBC_FIELDS:
        try:
            values[key] = float(widgets[f"lab_{key}"].get())
        except (ValueError, KeyError):
            values[key] = None
    for key, label, options, cutset in LAB_PS_FIELDS:
        v = widgets.get(f"lab_{key}")
        v = v.get() if v is not None else "Not selected"
        values[key] = None if v == "Not selected" else v
    return values

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

# Colors
CLR_BG       = "#0f0f1a"
CLR_CARD     = "#1a1a2e"
CLR_BORDER   = "#2e2e50"
CLR_GREEN    = "#0F6E56"
CLR_GREEN2   = "#1a9070"
CLR_TEXT     = "#e0e0e0"
CLR_MUTED    = "#888888"
CLR_RED      = "#c0392b"
CLR_SUCCESS  = "#27ae60"
CLR_WHITE    = "#ffffff"

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class IDAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.lang      = "en"
        self.view_mode = "patient"   # "patient" or "doctor"
        self.field_widgets = {}
        self.last_row   = None       # for per-patient chart

        self.title("IDA AI Screening Tool")
        self.geometry("920x780")
        self.minsize(700, 600)
        self.configure(bg=CLR_BG)

        try:
            import os
            self._logo_img = tk.PhotoImage(file=os.path.join(os.path.dirname(__file__), "assets", "logo_64.png"))
            self.iconphoto(True, self._logo_img)
        except Exception:
            pass  # missing/unsupported icon shouldn't block the app from starting

        self._build_ui()

    # -----------------------------------------------------------------------
    @property
    def L(self):
        return LANG[self.lang]

    def toggle_lang(self):
        self.lang = "hi" if self.lang == "en" else "en"
        self._rebuild()

    def toggle_view(self):
        if self.view_mode == "patient":
            # Trying to enter Doctor View -> ask for password
            pw = simpledialog.askstring(
                "Doctor Login / डॉक्टर लॉगिन",
                "Enter doctor password / डॉक्टर पासवर्ड डालें:",
                show="*", parent=self
            )
            if pw is None:
                return  # user cancelled
            if pw != DOCTOR_PASSWORD:
                messagebox.showerror(
                    "Incorrect Password / गलत पासवर्ड",
                    "The password you entered is incorrect."
                )
                return
            self.view_mode = "doctor"
        else:
            self.view_mode = "patient"
        self._rebuild()

    def open_data_collection(self):
        pw = simpledialog.askstring(
            "Data Collection Login",
            "Enter data collection password:",
            show="*", parent=self
        )
        if pw is None:
            return
        if pw != DATA_ENTRY_PASSWORD:
            messagebox.showerror("Incorrect Password", "The password you entered is incorrect.")
            return
        self._build_data_collection_window()

    def _build_data_collection_window(self):
        win = tk.Toplevel(self)
        win.title("Patient Data Collection")
        win.geometry("920x760")
        win.configure(bg=CLR_BG)

        _engine, _is_cloud = get_db_engine()
        if _is_cloud:
            status_text = "☁ Connected to cloud database — records are stored permanently."
            status_color = "#4ade80"
        else:
            status_text = ("⚠ No cloud database configured — using local storage only "
                            "(patient_records.db). See README to connect a permanent database. "
                            "Export records regularly as a backup.")
            status_color = "#c9a227"
        tk.Label(win, text=status_text, font=("Segoe UI", 8), fg=status_color, bg=CLR_BG,
                 wraplength=740, justify="left").pack(fill="x", padx=14, pady=(10,4))

        # Scrollable form
        outer = tk.Frame(win, bg=CLR_BG)
        outer.pack(fill="both", expand=True, padx=14, pady=6)
        canvas = tk.Canvas(outer, bg=CLR_BG, bd=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        form = tk.Frame(canvas, bg=CLR_BG)
        canvas.create_window((0,0), window=form, anchor="nw")
        form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        de_widgets = {}

        def add_section(text):
            f = tk.Frame(form, bg="#16162a")
            f.pack(fill="x", pady=(10,4))
            tk.Label(f, text=text, font=("Segoe UI",9,"bold"), fg=CLR_MUTED, bg="#16162a",
                     anchor="w").pack(fill="x", padx=10, pady=6)

        def add_text_entry(label_text, key):
            row = tk.Frame(form, bg=CLR_BG)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label_text, width=30, anchor="w", font=("Segoe UI",9),
                     fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
            var = tk.StringVar()
            tk.Entry(row, textvariable=var, width=35, bg=CLR_CARD, fg=CLR_TEXT,
                      insertbackground=CLR_TEXT, relief="flat").pack(side="left")
            de_widgets[key] = var

        add_section("Patient Identification")
        add_text_entry("Patient ID / Registration Number", "patient_id")
        add_text_entry("Patient Name (optional)", "patient_name")
        add_text_entry("Phone Number (optional)", "patient_phone")

        consent_row = tk.Frame(form, bg=CLR_BG)
        consent_row.pack(fill="x", padx=10, pady=(4,8))
        consent_var = tk.BooleanVar(value=False)
        tk.Checkbutton(consent_row, variable=consent_var, bg=CLR_BG, selectcolor=CLR_CARD,
                        activebackground=CLR_BG, fg=CLR_TEXT,
                        text="  Patient / guardian has given consent to record and use this health\n  information for the IDA screening study.",
                        justify="left", anchor="w").pack(side="left")
        de_widgets["consent_given"] = consent_var

        add_section("Demographics")
        for col_name, key, kind, opts in DEMO_FIELDS:
            row = tk.Frame(form, bg=CLR_BG)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=self.L[key], width=30, anchor="w", font=("Segoe UI",9),
                     fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
            if kind == "age":
                var = tk.StringVar(value="30")
                ttk.Spinbox(row, from_=1, to=110, textvariable=var, width=12).pack(side="left")
            else:
                display = [f"{c} - {t}" for c,t in opts]
                var = tk.StringVar(value=display[0])
                ttk.Combobox(row, textvariable=var, values=display, state="readonly", width=32).pack(side="left")
            de_widgets[col_name] = var

        add_section("Signs & Symptoms")
        for col_name, label, kind, opts in SYMPTOM_FIELDS:
            row = tk.Frame(form, bg=CLR_BG)
            row.pack(fill="x", padx=10, pady=5)
            tk.Label(row, text=label, anchor="w", font=("Segoe UI",9,"bold"),
                     fg=CLR_TEXT, bg=CLR_BG, wraplength=680, justify="left").pack(fill="x")
            display = [f"{c} - {t}" for c,t in opts]
            var = tk.StringVar(value=display[0])
            cb = ttk.Combobox(row, textvariable=var, values=display, state="readonly", width=95)
            cb.pack(fill="x", pady=(2,0))
            de_widgets[col_name] = var

        add_section("Laboratory Report — CBC & Peripheral Smear")
        row = tk.Frame(form, bg=CLR_BG)
        row.pack(fill="x", padx=10, pady=3)
        tk.Label(row, text="Lab report available now?", width=30, anchor="w", font=("Segoe UI",9),
                 fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
        lab_avail_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, variable=lab_avail_var, bg=CLR_BG,
                        text="  (leave unchecked if patient hasn't been tested yet)",
                        fg=CLR_TEXT, selectcolor=CLR_CARD, activebackground=CLR_BG).pack(side="left")
        de_widgets["lab_available"] = lab_avail_var

        tk.Label(form, text="Complete Blood Count (CBC)", font=("Segoe UI",9,"bold"),
                 fg=CLR_TEXT, bg=CLR_BG, anchor="w").pack(fill="x", padx=10, pady=(8,2))
        for key, label, unit, default, lo, hi, step, op, cut in LAB_CBC_FIELDS:
            row = tk.Frame(form, bg=CLR_BG)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{label} ({unit})", width=32, anchor="w", font=("Segoe UI",9),
                     fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
            var = tk.StringVar(value=str(default))
            tk.Entry(row, textvariable=var, width=14, bg=CLR_CARD, fg=CLR_TEXT,
                      insertbackground=CLR_TEXT, relief="flat").pack(side="left")
            de_widgets[f"lab_{key}"] = var

        tk.Label(form, text="Peripheral Smear (PS)", font=("Segoe UI",9,"bold"),
                 fg=CLR_TEXT, bg=CLR_BG, anchor="w").pack(fill="x", padx=10, pady=(8,2))
        tk.Label(form, text="Note: Anisocytosis, Poikilocytosis & Pencil cells are only written in\n"
                             "the report when PRESENT — if not mentioned at all, select Absent\n"
                             "(don't leave as 'Not selected').",
                 font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_BG, anchor="w", justify="left").pack(fill="x", padx=10, pady=(0,4))
        for key, label, options, cutset in LAB_PS_FIELDS:
            row = tk.Frame(form, bg=CLR_BG)
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, width=32, anchor="w", font=("Segoe UI",9),
                     fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
            var = tk.StringVar(value="Not selected")
            ttk.Combobox(row, textvariable=var, state="readonly", width=20,
                         values=["Not selected"] + options).pack(side="left")
            de_widgets[f"lab_{key}"] = var

        tk.Label(form, text="The laboratory verdict (Suggestive / Not suggestive of IDA) is calculated "
                             "automatically from these values against Dr. Dixit's cut-off table — "
                             "no manual selection needed.", font=("Segoe UI",8), fg="#8a8aa0", bg=CLR_BG,
                 anchor="w", justify="left", wraplength=680).pack(fill="x", padx=10, pady=(6,6))

        tk.Frame(form, height=10, bg=CLR_BG).pack()

        add_section("Clinician's Assessment (Human Judgement)")
        tk.Label(form, text="This system is a screening aid, not a replacement for clinical judgement. "
                             "If you have your own impression of this patient based on examination — "
                             "independent of the AI prediction — record it here.",
                 font=("Segoe UI",8), fg="#8a8aa0", bg=CLR_BG, anchor="w", justify="left",
                 wraplength=680).pack(fill="x", padx=10, pady=(0,6))
        row = tk.Frame(form, bg=CLR_BG)
        row.pack(fill="x", padx=10, pady=3)
        tk.Label(row, text="Clinician's own assessment", width=30, anchor="w", font=("Segoe UI",9),
                 fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
        clinician_var = tk.StringVar(value="Not assessed")
        ttk.Combobox(row, textvariable=clinician_var, state="readonly", width=40, values=[
            "Not assessed", "IDA Positive (Clinical Judgement + Lab Reports)",
            "IDA Negative (Clinical Judgement + Lab Reports)", "Uncertain / Needs further workup"
        ]).pack(side="left")
        de_widgets["clinician_assessment"] = clinician_var

        row = tk.Frame(form, bg=CLR_BG)
        row.pack(fill="x", padx=10, pady=3)
        tk.Label(row, text="Clinician's notes (optional)", width=30, anchor="w", font=("Segoe UI",9),
                 fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
        clinician_notes_var = tk.StringVar(value="")
        tk.Entry(row, textvariable=clinician_notes_var, width=40, bg=CLR_CARD, fg=CLR_TEXT,
                  insertbackground=CLR_TEXT, relief="flat").pack(side="left")
        de_widgets["clinician_notes"] = clinician_notes_var

        tk.Frame(form, height=10, bg=CLR_BG).pack()

        # Footer buttons
        footer = tk.Frame(win, bg=CLR_BG, pady=8)
        footer.pack(fill="x", padx=14)

        def do_save():
            pid = de_widgets["patient_id"].get().strip()
            if not pid:
                messagebox.showerror("Missing Patient ID", "Patient ID is required.")
                return
            if not de_widgets["consent_given"].get():
                messagebox.showerror("Consent Required",
                    "Please confirm patient/guardian consent (checkbox under Patient Identification) before saving.")
                return
            dup_records = patient_id_exists(pid)
            if dup_records:
                proceed = messagebox.askyesno(
                    "Duplicate Patient ID",
                    f"Patient ID '{pid}' already has {len(dup_records)} record(s) saved "
                    f"(most recent: {dup_records[-1].entry_timestamp}).\n\n"
                    "If this is a follow-up visit for the same patient, click No and use "
                    "'Update Lab Report' instead.\n\n"
                    "Save as a new record anyway (genuinely a different patient)?")
                if not proceed:
                    return
            try:
                sym_vals = {}
                for col_name, *_ in DEMO_FIELDS:
                    raw = de_widgets[col_name].get()
                    sym_vals[col_name] = int(raw) if col_name == "Age" else int(raw.split(" - ")[0])
                for col_name, *_ in SYMPTOM_FIELDS:
                    sym_vals[col_name] = int(de_widgets[col_name].get().split(" - ")[0])

                row_df = pd.DataFrame([[sym_vals[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                if USES_SCALED and scaler:
                    row_df = pd.DataFrame(scaler.transform(row_df), columns=FEATURE_COLUMNS)
                ai_pred = model.predict(row_df)[0]
                ai_proba = model.predict_proba(row_df)[0][1]
                ai_pred_label = "IDA Positive" if ai_pred == 1 else "IDA Negative"

                lab_vals = collect_lab_panel_values(de_widgets)
                if de_widgets["lab_available"].get():
                    lab_confirmed, _ = classify_lab_verdict(lab_vals, gender=sym_vals.get("Gender"))
                else:
                    lab_confirmed = "Pending — awaiting lab report"
                    lab_vals = {}  # don't store placeholder defaults if lab wasn't actually done

                # Automatic Socioeconomic (Kuppuswamy) classification, calculated
                # the moment demographics are entered (Dr. Dixit's modification #2)
                kup_score, kup_class = calculate_kuppuswamy(sym_vals["Education"], sym_vals["Occupation"], sym_vals["Income"])

                save_patient_record(pid, de_widgets["patient_name"].get().strip(),
                                     sym_vals, lab_vals, lab_confirmed,
                                     ai_pred_label, round(ai_proba*100,1),
                                     kup_score, kup_class,
                                     de_widgets["patient_phone"].get().strip() or None,
                                     de_widgets["clinician_assessment"].get(),
                                     de_widgets["clinician_notes"].get().strip() or None,
                                     de_widgets["consent_given"].get())
                messagebox.showinfo("Saved", f"Record saved for Patient ID: {pid}\n\n"
                                              f"Lab-confirmed: {lab_confirmed}\n\n"
                                              f"Socioeconomic Status (Kuppuswamy): {kup_class}\n"
                                              f"Score: {kup_score} / 29")
            except ValueError as e:
                messagebox.showerror("Input Error", f"Please check your lab values are numbers.\n{e}")

        def do_view_records():
            rwin = tk.Toplevel(win)
            rwin.title("Saved Patient Records")
            rwin.geometry("760x480")
            rwin.configure(bg=CLR_BG)

            view_mode = {"showing_hidden": False}

            cols = ("ID","Patient ID","Name","Timestamp","Lab Confirmed")
            tree = ttk.Treeview(rwin, columns=cols, show="headings")
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c, width=110)
            tree.pack(fill="both", expand=True, padx=10, pady=(10,4))

            status_lbl = tk.Label(rwin, text="", font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_BG)
            status_lbl.pack(anchor="w", padx=10)

            def refresh_tree():
                for row in tree.get_children():
                    tree.delete(row)
                records = load_all_records(active_only=not view_mode["showing_hidden"])
                for r in records:
                    tree.insert("", "end", values=r)
                status_lbl.configure(
                    text=f"Showing {'HIDDEN' if view_mode['showing_hidden'] else 'active'} records ({len(records)})"
                )

            def get_selected_id():
                sel = tree.selection()
                if not sel:
                    messagebox.showinfo("No selection", "Please select a record first.")
                    return None
                return int(tree.item(sel[0])["values"][0])

            def do_hide():
                rid = get_selected_id()
                if rid is None: return
                hide_record(rid)
                refresh_tree()

            def do_restore():
                rid = get_selected_id()
                if rid is None: return
                restore_record(rid)
                refresh_tree()

            def do_delete():
                rid = get_selected_id()
                if rid is None: return
                if messagebox.askyesno("Confirm Permanent Delete",
                                        f"Permanently delete record #{rid}?\nThis CANNOT be undone."):
                    delete_record_permanently(rid)
                    refresh_tree()

            def toggle_view():
                view_mode["showing_hidden"] = not view_mode["showing_hidden"]
                toggle_btn.configure(text="👁 Show Active Records" if view_mode["showing_hidden"] else "🙈 Show Hidden Records")
                refresh_tree()

            def do_export():
                from tkinter import filedialog
                path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files","*.xlsx"), ("CSV files","*.csv")],
                    initialfile="patient_records_export.xlsx")
                if not path:
                    return
                if path.lower().endswith(".csv"):
                    n = export_all_records_csv(path)
                else:
                    n = export_all_records_excel(path)
                messagebox.showinfo("Exported", f"{n} records exported to:\n{path}\n\n"
                                                 f"Lab values and AI/clinical verdicts are placed as the "
                                                 f"last columns.")

            btn_row = tk.Frame(rwin, bg=CLR_BG)
            btn_row.pack(fill="x", padx=10, pady=(4,10))

            tk.Button(btn_row, text="🙈 Hide Selected", command=do_hide,
                      bg=CLR_CARD, fg=CLR_TEXT, relief="flat", padx=8, pady=6).pack(side="left", padx=(0,4))
            tk.Button(btn_row, text="♻ Restore Selected", command=do_restore,
                      bg=CLR_CARD, fg=CLR_TEXT, relief="flat", padx=8, pady=6).pack(side="left", padx=4)
            tk.Button(btn_row, text="🗑 Delete Selected", command=do_delete,
                      bg="#4a1f1f", fg=CLR_TEXT, relief="flat", padx=8, pady=6).pack(side="left", padx=4)
            toggle_btn = tk.Button(btn_row, text="🙈 Show Hidden Records", command=toggle_view,
                      bg=CLR_CARD, fg=CLR_TEXT, relief="flat", padx=8, pady=6)
            toggle_btn.pack(side="left", padx=4)

            def do_update_lab():
                rid = get_selected_id()
                if rid is None: return
                rec = get_full_record(rid)
                if rec is None:
                    messagebox.showerror("Not found", "Could not load this record.")
                    return
                try:
                    existing_lab = json.loads(rec["lab_panel_json"] or "{}")
                except Exception:
                    existing_lab = {}

                dlg = tk.Toplevel(rwin)
                dlg.title(f"Update Lab Report — {rec['patient_id']}")
                dlg.configure(bg=CLR_BG)
                dlg.geometry("640x640")

                canvas = tk.Canvas(dlg, bg=CLR_BG, highlightthickness=0)
                scrollbar = ttk.Scrollbar(dlg, orient="vertical", command=canvas.yview)
                inner = tk.Frame(canvas, bg=CLR_BG)
                inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
                canvas.create_window((0,0), window=inner, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)
                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")

                fields = {}

                def add_num_row(key, label, unit, default):
                    row = tk.Frame(inner, bg=CLR_BG)
                    row.pack(fill="x", padx=14, pady=3)
                    tk.Label(row, text=f"{label} ({unit})", width=32, anchor="w", font=("Segoe UI",9),
                             fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
                    start = existing_lab.get(key, default)
                    var = tk.StringVar(value=str(start))
                    tk.Entry(row, textvariable=var, width=14, bg=CLR_CARD, fg=CLR_TEXT,
                              insertbackground=CLR_TEXT, relief="flat").pack(side="left")
                    fields[key] = var

                def add_cat_row(key, label, options):
                    row = tk.Frame(inner, bg=CLR_BG)
                    row.pack(fill="x", padx=14, pady=3)
                    tk.Label(row, text=label, width=32, anchor="w", font=("Segoe UI",9),
                             fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
                    start = existing_lab.get(key, "Not selected")
                    var = tk.StringVar(value=start if start in options else "Not selected")
                    ttk.Combobox(row, textvariable=var, state="readonly", width=18,
                                 values=["Not selected"] + options).pack(side="left")
                    fields[key] = var

                tk.Label(inner, text="Complete Blood Count (CBC)", font=("Segoe UI",9,"bold"),
                         fg=CLR_TEXT, bg=CLR_BG, anchor="w").pack(fill="x", padx=14, pady=(10,2))
                for key, label, unit, default, lo, hi, step, op, cut in LAB_CBC_FIELDS:
                    add_num_row(key, label, unit, default)

                tk.Label(inner, text="Peripheral Smear (PS)", font=("Segoe UI",9,"bold"),
                         fg=CLR_TEXT, bg=CLR_BG, anchor="w").pack(fill="x", padx=14, pady=(10,2))
                tk.Label(inner, text="Note: Anisocytosis, Poikilocytosis & Pencil cells are only written in\n"
                                     "the report when PRESENT — if not mentioned at all, select Absent\n"
                                     "(don't leave as 'Not selected').",
                         font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_BG, anchor="w", justify="left").pack(fill="x", padx=14, pady=(0,4))
                for key, label, options, cutset in LAB_PS_FIELDS:
                    add_cat_row(key, label, options)

                def save_lab():
                    lab_vals = {}
                    for key, label, unit, default, lo, hi, step, op, cut in LAB_CBC_FIELDS:
                        try:
                            lab_vals[key] = float(fields[key].get())
                        except ValueError:
                            messagebox.showerror("Invalid input", f"'{label}' must be a number.")
                            return
                    for key, label, options, cutset in LAB_PS_FIELDS:
                        v = fields[key].get()
                        lab_vals[key] = None if v == "Not selected" else v
                    verdict, _ = classify_lab_verdict(lab_vals, gender=rec.get("Gender"))
                    update_lab_values(rid, lab_vals, verdict)
                    messagebox.showinfo("Saved", f"Lab report saved.\nResult: {verdict}")
                    dlg.destroy()
                    refresh_tree()

                tk.Button(inner, text="💾 Save Lab Report", command=save_lab,
                          bg=CLR_GREEN, fg=CLR_WHITE, relief="flat", padx=10, pady=8
                          ).pack(fill="x", padx=14, pady=16)

            tk.Button(btn_row, text="🧪 Update Lab Report", command=do_update_lab,
                      bg=CLR_CARD, fg=CLR_TEXT, relief="flat", padx=8, pady=6).pack(side="left", padx=4)

            tk.Button(rwin, text="⬇ Export All Records (Excel/CSV)", command=do_export,
                      bg=CLR_GREEN, fg=CLR_WHITE, relief="flat", padx=10, pady=6).pack(pady=(0,10))

            refresh_tree()

        tk.Button(footer, text="💾 Save Patient Record", font=("Segoe UI",10,"bold"),
                  bg=CLR_GREEN, fg=CLR_WHITE, relief="flat", padx=14, pady=8,
                  cursor="hand2", command=do_save).pack(side="left")
        tk.Button(footer, text="📄 View / Export Records", font=("Segoe UI",9),
                  bg=CLR_CARD, fg=CLR_TEXT, relief="flat", padx=10, pady=8,
                  cursor="hand2", command=do_view_records).pack(side="left", padx=8)

    def _rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.field_widgets = {}
        self._build_ui()

    # -----------------------------------------------------------------------
    def _build_ui(self):
        L = self.L

        # ── Top bar ──────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=CLR_BG)
        topbar.pack(fill="x", padx=14, pady=(10, 0))

        tk.Button(topbar, text=L["lang_btn"], font=("Segoe UI", 9),
                  bg=CLR_CARD, fg=CLR_TEXT, relief="flat", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self.toggle_lang).pack(side="right", padx=(6,0))

        view_lbl = L["view_btn_doc"] if self.view_mode == "patient" else L["view_btn_pat"]
        tk.Button(topbar, text=view_lbl, font=("Segoe UI", 9),
                  bg=CLR_CARD, fg=CLR_TEXT, relief="flat", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self.toggle_view).pack(side="right")

        tk.Button(topbar, text="📋 Data Collection", font=("Segoe UI", 9),
                  bg=CLR_CARD, fg=CLR_TEXT, relief="flat", bd=0,
                  padx=10, pady=5, cursor="hand2",
                  command=self.open_data_collection).pack(side="right", padx=(0,6))

        # ── Header ───────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=CLR_GREEN, pady=16)
        hdr.pack(fill="x", padx=14, pady=(8, 0))
        title_row = tk.Frame(hdr, bg=CLR_GREEN)
        title_row.pack()
        if getattr(self, "_logo_img", None):
            tk.Label(title_row, image=self._logo_img, bg=CLR_GREEN).pack(side="left", padx=(0, 8))
        else:
            tk.Label(title_row, text="🩸", font=("Segoe UI", 15), bg=CLR_GREEN, fg=CLR_WHITE).pack(side="left", padx=(0, 6))
        tk.Label(title_row, text=L["title"],
                 font=("Segoe UI", 15, "bold"),
                 fg=CLR_WHITE, bg=CLR_GREEN).pack(side="left")
        tk.Label(hdr, text=L["subtitle"],
                 font=("Segoe UI", 9),
                 fg="#c8ede5", bg=CLR_GREEN).pack()

        # ── Stats row (Doctor only) ───────────────────────────────────────
        if self.view_mode == "doctor":
            stats = tk.Frame(self, bg=CLR_BG)
            stats.pack(fill="x", padx=14, pady=(8,0))
            for lbl, val, sub in [
                ("Model Accuracy", "99.2%", "on test records"),
                ("AUC-ROC Score",  "0.999", "near-perfect"),
                ("Training Data",  "20,000","patient records"),
            ]:
                card = tk.Frame(stats, bg=CLR_CARD, bd=0, relief="flat",
                                padx=14, pady=10)
                card.pack(side="left", expand=True, fill="x", padx=4)
                tk.Label(card, text=lbl, font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_CARD).pack(anchor="w")
                tk.Label(card, text=val, font=("Segoe UI",18,"bold"), fg=CLR_WHITE, bg=CLR_CARD).pack(anchor="w")
                tk.Label(card, text=sub, font=("Segoe UI",8), fg=CLR_MUTED, bg=CLR_CARD).pack(anchor="w")

        # ── Scrollable form ───────────────────────────────────────────────
        outer = tk.Frame(self, bg=CLR_BG)
        outer.pack(fill="both", expand=True, padx=14, pady=8)

        canvas  = tk.Canvas(outer, bg=CLR_BG, bd=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.form = tk.Frame(canvas, bg=CLR_BG)
        canvas.create_window((0,0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-1*(e.delta/120)), "units"))

        # Demographics
        self._section(L["demo_head"])
        for col, key, kind, opts in DEMO_FIELDS:
            self._field(col, L[key], kind, opts)

        # Symptoms
        self._section(L["symp_head"])
        for col, label, kind, opts in SYMPTOM_FIELDS:
            self._field(col, label, kind, opts, wide=True)

        tk.Frame(self.form, height=10, bg=CLR_BG).pack()

        # ── Footer ───────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=CLR_BG, pady=8)
        footer.pack(fill="x", padx=14)

        tk.Button(footer, text=L["predict_btn"],
                  font=("Segoe UI", 11, "bold"),
                  bg=CLR_GREEN, fg=CLR_WHITE, relief="flat",
                  padx=20, pady=10, cursor="hand2",
                  command=self.predict).pack(side="left")

        # Chart picker (Doctor only)
        if self.view_mode == "doctor":
            tk.Label(footer, text=L["chart_lbl"],
                     font=("Segoe UI",9), fg=CLR_MUTED,
                     bg=CLR_BG).pack(side="left", padx=(16,4))
            self.chart_var = tk.StringVar(value=L["chart_fi"])
            chart_dd = ttk.Combobox(footer, textvariable=self.chart_var,
                                    state="readonly", width=18,
                                    values=[L["chart_fi"], L["chart_shap"],
                                            L["chart_cmp"], L["chart_pat"]])
            chart_dd.pack(side="left", padx=2)
            tk.Button(footer, text=L["view_chart"],
                      font=("Segoe UI",9), bg=CLR_CARD, fg=CLR_TEXT,
                      relief="flat", padx=8, pady=5, cursor="hand2",
                      command=self.show_chart).pack(side="left", padx=4)

        # Result label
        self.result_var = tk.StringVar(
            value="Fill in the form and click 'Run AI Prediction'")
        self.result_lbl = tk.Label(self, textvariable=self.result_var,
                                   font=("Segoe UI", 13, "bold"),
                                   fg=CLR_MUTED, bg=CLR_BG, wraplength=760,
                                   justify="center", pady=10)
        self.result_lbl.pack(fill="x", padx=14)

        # Advice label
        self.advice_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.advice_var,
                 font=("Segoe UI",9), fg=CLR_MUTED,
                 bg=CLR_BG, wraplength=760, justify="center").pack()

        # Kuppuswamy Socioeconomic Scale (Doctor View only)
        if self.view_mode == "doctor":
            self.kuppuswamy_var = tk.StringVar(value="")
            kup_frame = tk.Frame(self, bg="#16162a")
            kup_frame.pack(fill="x", padx=14, pady=(8,0))
            tk.Label(kup_frame, text="📋 Socioeconomic Status — Modified Kuppuswamy Scale",
                     font=("Segoe UI",8,"bold"), fg=CLR_MUTED, bg="#16162a").pack(anchor="w", padx=10, pady=(6,0))
            tk.Label(kup_frame, textvariable=self.kuppuswamy_var,
                     font=("Segoe UI",11,"bold"), fg=CLR_TEXT, bg="#16162a",
                     justify="left", anchor="w").pack(anchor="w", padx=10, pady=(0,6))
        else:
            self.kuppuswamy_var = None

        # Disclaimer
        tk.Label(self, text=L["disclaimer"],
                 font=("Segoe UI",8), fg="#444466",
                 bg=CLR_BG, wraplength=760).pack(pady=(4,10))

    # -----------------------------------------------------------------------
    def _section(self, text):
        f = tk.Frame(self.form, bg="#16162a")
        f.pack(fill="x", pady=(10,4))
        tk.Label(f, text=text, font=("Segoe UI",9,"bold"),
                 fg=CLR_MUTED, bg="#16162a",
                 anchor="w").pack(fill="x", padx=10, pady=6)

    def _field(self, col_name, label, kind, options, wide=False):
        row = tk.Frame(self.form, bg=CLR_BG)
        row.pack(fill="x", padx=10, pady=3 if not wide else 5)
        if kind == "age":
            tk.Label(row, text=label, width=36, anchor="w",
                     font=("Segoe UI",9), fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
            var = tk.StringVar(value="30")
            w   = ttk.Spinbox(row, from_=1, to=110, textvariable=var, width=12)
            w.pack(side="left")
            self.field_widgets[col_name] = ("age", var)
        elif wide:
            # Symptom rows: stack label above the dropdown and use the full
            # window width so the English translation is never cut off.
            tk.Label(row, text=label, anchor="w", font=("Segoe UI",9,"bold"),
                     fg=CLR_TEXT, bg=CLR_BG, wraplength=680, justify="left").pack(fill="x")
            display = [f"{c} - {t}" for c,t in options]
            var = tk.StringVar(value=display[0])
            w   = ttk.Combobox(row, textvariable=var, values=display,
                                state="readonly", width=95)
            w.pack(fill="x", pady=(2,0))
            self.field_widgets[col_name] = ("choice", var)
        else:
            tk.Label(row, text=label, width=36, anchor="w",
                     font=("Segoe UI",9), fg=CLR_TEXT, bg=CLR_BG).pack(side="left")
            display = [f"{c} - {t}" for c,t in options]
            var = tk.StringVar(value=display[0])
            w   = ttk.Combobox(row, textvariable=var, values=display,
                                state="readonly", width=44)
            w.pack(side="left")
            self.field_widgets[col_name] = ("choice", var)

    # -----------------------------------------------------------------------
    def _collect(self):
        vals = {}
        for col, (kind, var) in self.field_widgets.items():
            raw = var.get()
            if kind == "age":
                try:
                    v = int(raw)
                except ValueError:
                    raise ValueError(f"Age must be a number. Got: '{raw}'")
                if not 1 <= v <= 110:
                    raise ValueError("Age must be between 1 and 110.")
                vals[col] = v
            else:
                vals[col] = int(raw.split(" - ")[0])
        return vals

    def predict(self):
        L = self.L
        try:
            vals = self._collect()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        row = pd.DataFrame([[vals[c] for c in FEATURE_COLUMNS]],
                            columns=FEATURE_COLUMNS)
        if USES_SCALED and scaler:
            row = pd.DataFrame(scaler.transform(row), columns=FEATURE_COLUMNS)
        self.last_row = row

        pred  = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        conf  = proba*100 if pred==1 else (1-proba)*100

        if pred == 1:
            self.result_var.set(f"{L['present']}   ({L['conf']}: {conf:.1f}%)")
            self.result_lbl.configure(fg=CLR_RED)
            self.advice_var.set(L["advice_p"])
        else:
            self.result_var.set(f"{L['absent']}   ({L['conf']}: {conf:.1f}%)")
            self.result_lbl.configure(fg=CLR_SUCCESS)
            self.advice_var.set(L["advice_a"])

        if self.kuppuswamy_var is not None:
            kup_score, kup_class = calculate_kuppuswamy(
                vals["Education"], vals["Occupation"], vals["Income"]
            )
            self.kuppuswamy_var.set(f"{kup_class}   (Score: {kup_score} / 29)")

    # -----------------------------------------------------------------------
    def show_chart(self):
        if not hasattr(self, "chart_var"):
            return
        L    = self.L
        sel  = self.chart_var.get()
        fmap = {
            L["chart_fi"]:   "feature_importance.png",
            L["chart_shap"]: "shap_summary.png",
            L["chart_cmp"]:  "model_comparison.png",
        }

        if sel == L["chart_pat"]:
            # Per-patient SHAP chart
            if self.last_row is None:
                messagebox.showinfo("No prediction",
                    "Run a prediction first, then view this chart.")
                return
            self._show_patient_chart()
            return

        path = fmap.get(sel)
        if not path or not os.path.exists(path):
            messagebox.showinfo("Not found",
                f"'{path}' not found. Run train_model.py to generate it.")
            return

        # Open chart in a new window
        win = tk.Toplevel(self)
        win.title(sel)
        win.configure(bg=CLR_BG)
        img  = Image.open(path)
        img.thumbnail((820, 700))
        photo = ImageTk.PhotoImage(img)
        lbl   = tk.Label(win, image=photo, bg=CLR_BG)
        lbl.image = photo
        lbl.pack(padx=10, pady=10)

    def _show_patient_chart(self):
        L = self.L
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(self.last_row)
            if isinstance(sv, list):
                sv_p = sv[1][0]
            elif sv.ndim == 3:
                sv_p = sv[0, :, 1]
            else:
                sv_p = sv[0]

            contrib = pd.Series(sv_p, index=FEATURE_COLUMNS)
            contrib = contrib[contrib != 0].sort_values()

            if len(contrib) == 0:
                messagebox.showinfo("Chart", "All contributions are zero — model is uncertain.")
                return

            colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in contrib.values]
            fig, ax = plt.subplots(figsize=(7, max(3, len(contrib)*0.35)))
            fig.patch.set_facecolor("#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            contrib.plot(kind="barh", ax=ax, color=colors)
            ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Impact on prediction", color="white")
            ax.set_title(L["chart_pat"] + " — Symptom contributions", color="white")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            win   = tk.Toplevel(self)
            win.title(L["chart_pat"])
            win.configure(bg=CLR_BG)
            img   = Image.open(buf)
            photo = ImageTk.PhotoImage(img)
            lbl   = tk.Label(win, image=photo, bg=CLR_BG)
            lbl.image = photo
            lbl.pack(padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("Chart Error", str(e))


if __name__ == "__main__":
    app = IDAApp()
    app.mainloop()
