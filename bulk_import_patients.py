"""
bulk_import_patients.py
========================
One-off script to bulk-import the hospital's existing "IDA_Data_Collection.xlsx"
sheet (real patients, e.g. ~1598 rows) into the app's database.

WHAT THIS DOES
--------------
The sheet only reliably contains: Record_ID, Age, Gender, Hemoglobin, and a
clinician-determined Outcome (Yes/No). It does NOT contain CBC (MCV/MCH),
Peripheral Smear, or the demographic/symptom fields the app's own form
collects. So each imported record gets:
  - patient_id   <- Record_ID from the sheet
  - Age, Gender  <- cleaned from the sheet
  - hemoglobin   <- cleaned from the sheet (may be blank if not recorded)
  - lab_confirmed_ida <- "Lab values not yet entered" (honest: we don't have
    the CBC/PS values our classify_lab_verdict() rule needs, only Hb)
  - clinician_assessment <- the hospital's own Outcome (Yes -> "IDA Positive
    (Clinical Judgement + Lab Reports)", No -> "IDA Negative (...)") — this
    preserves their diagnosis without pretending our app calculated it
  - Paleness, Fatigue, SOB, Headache, Dizziness, Infections, Appetite, Pica,
    ColdIntolerance <- DERIVED from the Hb value using Dr. Dixit's reference
    grading tables (since the sheet's own symptom columns for these are
    mostly free-text/unreliable, but Hb is reliably recorded). Grade codes
    match this app's own SYMPTOM_FIELDS 0-3 scale exactly. Infections gets
    a special override: wherever the sheet's "Cold Intolerance" column
    (which turned out to actually contain misplaced infection notes like
    "fever/cold/cough") mentions fever, cold, or cough, Infections is
    force-set to Grade 2 regardless of what Hb alone would suggest.
  - Everything else (Address, Education, Occupation, Income, the other 13
    symptoms, PS/CBC beyond Hb) is left NULL/blank — genuinely not
    available or not reliably derivable, not guessed.

Data cleaning handles common typos automatically (e.g. "37y"/"9month" for
Age, "11.4gm?%" for Hemoglobin, "femela"/"fe,male" for Gender). Rows with
a value that's too broken to safely parse (garbage age/Hb, ambiguous
Gender/Outcome) are written to patients_flagged_for_review.csv instead of
being imported — these need a human to look at the original sheet and
decide, rather than a script guessing.

HOW TO RUN
----------
    python bulk_import_patients.py <path_to_excel>              # DRY RUN
    python bulk_import_patients.py <path_to_excel> --apply       # writes to DB

Example:
    python bulk_import_patients.py IDA_Data_Collection.xlsx
    python bulk_import_patients.py IDA_Data_Collection.xlsx --apply

Uses the same DB connection resolution as recalculate_lab_verdicts.py:
SUPABASE_DB_URL environment variable -> .streamlit/secrets.toml -> local
patient_records.db SQLite fallback.
"""
import os
import sys
import re
import json
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, text


# ---------------------------------------------------------------------------
# 1. DB connection — identical resolution order to recalculate_lab_verdicts.py
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
            print("Connected to Supabase/Postgres database.")
            return engine
        except Exception as e:
            print(f"Could not connect using SUPABASE_DB_URL ({e}).")
            print("Falling back to local patient_records.db (SQLite).")

    engine = create_engine("sqlite:///patient_records.db")
    print("Connected to local patient_records.db (SQLite).")
    return engine


# ---------------------------------------------------------------------------
# 2. Cleaning functions — see module docstring for the reasoning behind
#    what counts as "safely parseable" vs "needs a human to look at it".
# ---------------------------------------------------------------------------
def clean_age(v):
    if pd.isna(v):
        return None, "ok"  # genuinely not recorded -> acceptable, store NULL
    s = str(v).strip().lower()
    m = re.match(r'^(\d+)\s*month', s)
    if m:
        return 0, "ok"  # under 1 year -> 0 completed years
    m = re.match(r'^(\d+\.?\d*)', s)
    if m:
        try:
            n = int(float(m.group(1)))
            if 0 < n < 120:
                return n, "ok"
            return None, f"age_out_of_range:{v}"
        except ValueError:
            pass
    return None, f"age_unparseable:{v}"


def clean_gender(v):
    if pd.isna(v):
        return None, "gender_missing"
    s = str(v).strip().lower()
    if s.startswith("m"):
        return 1, "ok"
    if s.startswith("f"):
        return 2, "ok"
    return None, f"gender_unparseable:{v}"


def clean_hb(v):
    if pd.isna(v):
        return None, "ok"  # genuinely not recorded -> acceptable, store NULL
    s = str(v).strip().lower()
    m = re.match(r'^(\d+\.?\d*)', s)
    if m:
        try:
            n = float(m.group(1))
            if 0 < n < 25:
                return round(n, 1), "ok"
            return None, f"hb_out_of_range:{v}"
        except ValueError:
            pass
    return None, f"hb_unparseable:{v}"


def clean_outcome(v):
    if pd.isna(v):
        return None, "outcome_missing"
    s = str(v).strip().lower()
    if s == "yes":
        return "IDA Positive (Clinical Judgement + Lab Reports)", "ok"
    if s == "no":
        return "IDA Negative (Clinical Judgement + Lab Reports)", "ok"
    return None, f"outcome_unparseable:{v}"


# ---------------------------------------------------------------------------
# Hb-based symptom severity derivation, per Dr. Dixit's reference tables.
# Since the sheet's own symptom columns are mostly free-text/messy, but Hb is
# reliably recorded, these 9 symptoms are DERIVED from the Hb value instead
# of read from their (unreliable) columns. Grades match this app's own
# SYMPTOM_FIELDS 0-3 codes exactly (same wording), so no further mapping is
# needed once the grade number is picked.
# ---------------------------------------------------------------------------
def pallor_grade(hb):
    """Dr. Dixit's Pallor table (distinct thresholds from the other 8)."""
    if hb is None:
        return None
    if hb >= 12:
        return 0
    if hb >= 10:
        return 0
    if hb >= 8:
        return 1
    if hb >= 7:
        return 2
    return 3


def standard_grade(hb):
    """Shared WHO-style Hb bucket used by Fatigue, SOB, Headache, Dizziness,
    Infections, Appetite, Pica, and Cold Intolerance."""
    if hb is None:
        return None
    if hb >= 12:
        return 0
    if hb >= 11:
        return 1
    if hb >= 8:
        return 2
    return 3


INFECTION_KEYWORDS = ("fever", "cold", "cough")


def derive_symptoms(hb, infection_note_text):
    """Returns a dict of the 9 Hb-derived symptom severities. `infection_note_text`
    is the raw free-text value from the sheet's "Cold Intolerance" column, which
    turned out to actually contain infection notes (fever/cold/cough) rather than
    real cold-intolerance data — per Dr. Dixit: wherever that text mentions
    fever/cold/cough, Infections is force-set to Grade 2 regardless of the
    Hb-derived value. Cold Intolerance itself is purely Hb-derived, since the
    real cold-intolerance data was never actually recorded in that column."""
    infections = standard_grade(hb)
    if isinstance(infection_note_text, str):
        note = infection_note_text.strip().lower()
        if any(k in note for k in INFECTION_KEYWORDS):
            infections = 2

    return {
        "Paleness": pallor_grade(hb),
        "Fatigue": standard_grade(hb),
        "SOB": standard_grade(hb),
        "Headache": standard_grade(hb),
        "Dizziness": standard_grade(hb),
        "Infections": infections,
        "Appetite": standard_grade(hb),
        "Pica": standard_grade(hb),
        "ColdIntolerance": standard_grade(hb),
    }


def clean_sheet(path):
    """Reads the hospital's Excel sheet and returns (clean_df, flagged_df).
    Assumes the same layout as IDA_Data_Collection.xlsx: header row is the
    3rd row (index 2), data starts on the 4th row (index 3), with Record_ID,
    Age, Gender in the first 3 columns and Outcome in the last column."""
    df_raw = pd.read_excel(path, sheet_name=0, header=None)
    cols = df_raw.iloc[2].fillna("").astype(str).tolist()
    data = df_raw.iloc[3:].reset_index(drop=True)
    data.columns = cols

    record_id_col = cols[0]
    age_col = cols[1]
    gender_col = cols[2]
    hb_col = next((c for c in cols if "aemoglobi" in c.lower() or "hemoglobi" in c.lower()), None)
    infection_note_col = cols[20]  # mislabeled "Cold Intolerance" column that actually holds fever/cold/cough notes
    outcome_col = cols[-1]

    if hb_col is None:
        raise ValueError("Could not find a Hemoglobin column in this sheet — check the file layout.")

    rows_clean, rows_flagged = [], []
    for _, row in data.iterrows():
        rid = row[record_id_col]
        if pd.isna(rid):
            continue
        patient_id = str(int(rid)) if isinstance(rid, (int, float)) else str(rid).strip()

        age, age_status = clean_age(row[age_col])
        gender, gender_status = clean_gender(row[gender_col])
        hb, hb_status = clean_hb(row[hb_col])
        outcome, outcome_status = clean_outcome(row[outcome_col])

        issues = [s for s in (age_status, gender_status, hb_status, outcome_status) if s != "ok"]
        record = {"patient_id": patient_id, "Age": age, "Gender": gender,
                  "hemoglobin": hb, "clinician_assessment": outcome, "issues": "; ".join(issues)}
        record.update(derive_symptoms(hb, row[infection_note_col]))
        (rows_flagged if issues else rows_clean).append(record)

    return pd.DataFrame(rows_clean), pd.DataFrame(rows_flagged)


# ---------------------------------------------------------------------------
# 3. Main import pass.
# ---------------------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    apply_changes = "--apply" in sys.argv

    if not args:
        print("Usage: python bulk_import_patients.py <path_to_excel> [--apply]")
        return
    excel_path = args[0]

    print(f"Reading and cleaning {excel_path} ...")
    clean_df, flagged_df = clean_sheet(excel_path)
    print(f"\n{len(clean_df) + len(flagged_df)} total rows in the sheet.")
    print(f"{len(clean_df)} rows are clean and ready to import.")
    print(f"{len(flagged_df)} rows have a genuinely ambiguous value and are SKIPPED "
          f"(saved to patients_flagged_for_review.csv for a human to check).\n")

    flagged_df.to_csv("patients_flagged_for_review.csv", index=False)

    engine = get_engine()
    metadata = MetaData()
    patients_table = Table("patients", metadata, autoload_with=engine)

    with engine.connect() as conn:
        existing_ids = {row[0] for row in conn.execute(text("SELECT patient_id FROM patients")).fetchall()}

    new_rows = clean_df[~clean_df["patient_id"].isin(existing_ids)]
    dup_rows = clean_df[clean_df["patient_id"].isin(existing_ids)]

    print(f"{len(dup_rows)} rows already exist in the database (same Patient ID) — will be SKIPPED, not overwritten.")
    print(f"{len(new_rows)} new rows will be inserted.\n")

    if not apply_changes:
        print("This was a DRY RUN — nothing was written to the database.")
        print("Review patients_flagged_for_review.csv, then re-run with:")
        print(f"    python bulk_import_patients.py {excel_path} --apply")
        return

    if len(new_rows) == 0:
        print("Nothing new to import.")
        return

    confirm = input(f"Type YES to insert these {len(new_rows)} new patient record(s) into the database: ")
    if confirm.strip() != "YES":
        print("Aborted — no changes written.")
        return

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    derived_symptom_cols = ["Paleness", "Fatigue", "SOB", "Headache", "Dizziness",
                             "Infections", "Appetite", "Pica", "ColdIntolerance"]
    inserted = 0
    with engine.begin() as conn:
        for _, r in new_rows.iterrows():
            lab_panel = {"hb": r["hemoglobin"]} if pd.notna(r["hemoglobin"]) else {}
            values = dict(
                patient_id=r["patient_id"],
                entry_timestamp=now_iso,
                is_active=1,
                Age=int(r["Age"]) if pd.notna(r["Age"]) else None,
                Gender=int(r["Gender"]) if pd.notna(r["Gender"]) else None,
                hemoglobin=float(r["hemoglobin"]) if pd.notna(r["hemoglobin"]) else None,
                lab_panel_json=json.dumps(lab_panel) if lab_panel else None,
                lab_confirmed_ida="Lab values not yet entered",
                clinician_assessment=r["clinician_assessment"] if pd.notna(r["clinician_assessment"]) else "Not assessed",
            )
            for col in derived_symptom_cols:
                values[col] = int(r[col]) if pd.notna(r.get(col)) else None
            conn.execute(patients_table.insert().values(**values))
            inserted += 1

    print(f"\nDone — inserted {inserted} new patient record(s).")
    print(f"{len(dup_rows)} duplicate Patient ID(s) skipped, {len(flagged_df)} flagged row(s) skipped "
          f"(see patients_flagged_for_review.csv).")


if __name__ == "__main__":
    main()
