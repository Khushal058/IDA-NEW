"""
recalculate_lab_verdicts.py
============================
One-off maintenance script.

WHY THIS EXISTS
----------------
Every patient record stores its FULL raw lab panel in the `lab_panel_json`
column (all CBC + Peripheral Smear values, exactly as entered) — this never
changes. But the `lab_confirmed_ida` column stores the *verdict text* that
was calculated at the time the record was saved, using whatever version of
`classify_lab_verdict()` was live in the app back then.

Since the CORE panel logic has been updated over time (Iron Profile removed
-> CBC-only core -> CBC+PS 10-parameter core), older records may have a
verdict computed with an OLDER rule than newer records. This script re-reads
every record's raw `lab_panel_json` and recalculates `lab_confirmed_ida`
using the CURRENT `classify_lab_verdict()` logic (copied verbatim from
web_app.py), so every patient — old and new — is judged by the exact same
rule. No new lab tests are needed; nothing except the verdict TEXT and the
quick-access mcv/mch/mchc/rdwcv columns is touched.

HOW TO RUN
----------
    python recalculate_lab_verdicts.py            # DRY RUN (default) — shows
                                                    # what WOULD change, writes nothing
    python recalculate_lab_verdicts.py --apply     # actually writes the updates

It looks for the database connection in this order:
  1. SUPABASE_DB_URL environment variable
  2. .streamlit/secrets.toml in this folder (same file Streamlit Cloud/local
     app uses)
  3. Falls back to the local patient_records.db SQLite file
"""
import os
import sys
import json

from sqlalchemy import create_engine, MetaData, Table, text

# ---------------------------------------------------------------------------
# 1. Get a DB connection — same resolution order the app uses.
# ---------------------------------------------------------------------------
def get_engine():
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            try:
                import tomllib
                with open(secrets_path, "rb") as f:
                    data = tomllib.load(f)
                db_url = data.get("SUPABASE_DB_URL")
            except ModuleNotFoundError:
                import toml
                data = toml.load(secrets_path)
                db_url = data.get("SUPABASE_DB_URL")

    if db_url:
        try:
            engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
            with engine.connect():
                pass
            print(f"Connected to Supabase/Postgres database.")
            return engine
        except Exception as e:
            print(f"Could not connect using SUPABASE_DB_URL ({e}).")
            print("Falling back to local patient_records.db (SQLite).")

    engine = create_engine("sqlite:///patient_records.db")
    print("Connected to local patient_records.db (SQLite).")
    return engine


# ---------------------------------------------------------------------------
# 2. Lab field definitions + classify_lab_verdict — verbatim copy of the
#    CURRENT logic in web_app.py, so this script always judges records by
#    whatever rule is live in the app today.
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
    """Verbatim copy of web_app.py's classify_lab_verdict — simple,
    deterministic CBC-OR-PS rule, no fractional "X/5"/"X/10" scoring."""
    hb, mcv, mch = lab_values.get("hb"), lab_values.get("mcv"), lab_values.get("mch")
    rbc_size, rbc_staining = lab_values.get("rbc_size"), lab_values.get("rbc_staining")

    hb_cutoff = 13.0 if gender == 1 else 12.0  # Male=13, Female/Other/unknown=12

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
        verdict = f"Not suggestive so far — {missing} not yet entered"

    return verdict, {"cbc_met": cbc_met, "ps_met": ps_met, "cbc_entered": cbc_entered, "ps_entered": ps_entered}


def _to_native(v):
    try:
        import numpy as np
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
    except ImportError:
        pass
    return v


# ---------------------------------------------------------------------------
# 3. Main recalculation pass.
# ---------------------------------------------------------------------------
def main():
    apply_changes = "--apply" in sys.argv

    engine = get_engine()
    metadata = MetaData()
    patients_table = Table("patients", metadata, autoload_with=engine)

    with engine.connect() as conn:
        rows = conn.execute(text(
            'SELECT record_id, patient_id, lab_panel_json, lab_confirmed_ida, "Gender" FROM patients'
        )).fetchall()

    print(f"\nFound {len(rows)} patient record(s) in the database.\n")

    changed = []
    unchanged = 0
    no_lab_data = 0

    for row in rows:
        record_id, patient_id, lab_json, old_verdict, gender = row
        if not lab_json:
            no_lab_data += 1
            continue

        try:
            lab_values = json.loads(lab_json)
        except (json.JSONDecodeError, TypeError):
            print(f"  [SKIP] record_id={record_id}: lab_panel_json could not be parsed.")
            continue

        new_verdict, detail = classify_lab_verdict(lab_values, gender=gender)

        if new_verdict != old_verdict:
            changed.append((record_id, patient_id, old_verdict, new_verdict, lab_values))
        else:
            unchanged += 1

    print(f"Records with no lab data yet: {no_lab_data}")
    print(f"Records whose verdict is UNCHANGED under the new rule: {unchanged}")
    print(f"Records whose verdict WOULD CHANGE under the new rule: {len(changed)}\n")

    if changed:
        print("-" * 90)
        for record_id, patient_id, old_v, new_v, _ in changed:
            print(f"  record_id={record_id}  patient_id={patient_id}")
            print(f"      OLD: {old_v}")
            print(f"      NEW: {new_v}")
        print("-" * 90)

    if not apply_changes:
        print("\nThis was a DRY RUN — nothing was written to the database.")
        print("Review the changes above, then re-run with:")
        print("    python recalculate_lab_verdicts.py --apply")
        return

    if not changed:
        print("\nNothing to update — database already matches the current rule.")
        return

    confirm = input(f"\nType YES to write these {len(changed)} update(s) to the database: ")
    if confirm.strip() != "YES":
        print("Aborted — no changes written.")
        return

    with engine.begin() as conn:
        for record_id, patient_id, old_v, new_v, lab_values in changed:
            conn.execute(
                patients_table.update().where(patients_table.c.record_id == record_id).values(
                    lab_confirmed_ida=new_v,
                    mcv=_to_native(lab_values.get("mcv")),
                    mch=_to_native(lab_values.get("mch")),
                    mchc=_to_native(lab_values.get("mchc")),
                    rdwcv=_to_native(lab_values.get("rdwcv")),
                )
            )

    print(f"\nDone — updated {len(changed)} record(s) to the current lab-verdict rule.")


if __name__ == "__main__":
    main()
