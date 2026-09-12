# Iron Deficiency Anaemia – AI Screening Tool

A full AI model + GUI (desktop and web) that predicts whether a patient
likely has Iron Deficiency Anaemia (IDA), using **only signs, symptoms,
and basic demographics — no blood test required**. Built from the
research dataset and proposal provided by Dr. Ruchita Dixit, Dr. Amol
Patil, and Dr. Smita Selot (SSIMS, Bhilai).

## How this matches the research proposal

The proposal's "Methodology for developing AI" section lists 8 steps.
This project implements every step that is actually a coding/AI task:

| Proposal step | Implemented as |
|---|---|
| Data Preprocessing | `train_model.py` Step 2 — missing values, duplicates, encoding check |
| Feature Engineering | `train_model.py` Step 3 — selects only demographics + symptoms, excludes lab values |
| Model Selection | `train_model.py` Step 5 — Decision Tree, Random Forest, SVM, and Neural Network are all trained and compared; best one is picked automatically |
| Model Training (train/val/test split + tuning) | `train_model.py` Steps 4 & 6 — 70/15/15 split + GridSearchCV hyperparameter tuning |
| Evaluation & Validation | `train_model.py` Steps 7 & 8 — accuracy, precision, recall, F1, AUC-ROC, plus 5-fold cross-validation |
| User Interface (mobile/web) | `gui_app.py` (desktop) and `web_app.py` (browser — also works on phones) |
| Explainability (feature importance + SHAP) | `train_model.py` Step 9 — both a feature-importance chart and a SHAP summary chart are generated |
| Clinical Validation | **Not included** — this requires real doctors reviewing live predictions, a research step for your faculty/co-investigators, not a coding task |
| Deployment into hospital EHR | **Not included** — this requires hospital IT infrastructure and compliance approval, outside the scope of a student project |

## Why no lab values are used as model input

The proposal's AIM is to diagnose IDA *based on signs and symptoms only*,
because many patients avoid blood tests. So although the dataset includes
lab values (Hemoglobin, Serum Iron, TIBC, Transferrin Saturation,
Ferritin), the AI model deliberately does **not** use them as inputs —
only age, gender, address, education, occupation, income, and 22
observable signs/symptoms are used. The lab values were only used
historically (by the doctors) to confirm the original diagnosis labels.

## What's in this folder

| File | What it is |
|---|---|
| `10K_individual_with_Iron_deficiency.xlsx` | Original data: 10,000 patients with IDA |
| `10K_individual_without_Iron_deficiency.xlsx` | Original data: 10,000 patients without IDA |
| `combined_dataset.csv` | Both files merged and shuffled (20,000 records) |
| `train_model.py` | Run this FIRST. Builds and compares 4 AI models, tunes and validates the best one, and saves it. |
| `gui_app.py` | Desktop GUI (Tkinter) — run AFTER training. |
| `web_app.py` | Web/mobile-friendly GUI (Streamlit) — run AFTER training. |
| `ida_model.joblib` | The final, best-performing trained model |
| `feature_list.joblib` | Exact list/order of inputs the model expects |
| `scaler.joblib` | Feature scaler (used only if the winning model needs scaled inputs) |
| `model_meta.joblib` | Records which algorithm won and whether it needs scaled inputs |
| `feature_importance.png` | Chart: which symptoms matter most |
| `shap_summary.png` | Chart: SHAP explainability — how each symptom pushes the prediction up or down |
| `model_comparison.png` | Chart: how all 4 candidate algorithms compared |
| `model_report.txt` | Full step-by-step methodology + performance report |
| `requirements.txt` | Python libraries needed |

## How to run it (step by step)

**1. Open this folder in VS Code.**

**2. Open a terminal in VS Code** (Terminal menu → New Terminal).

**3. Install the required libraries** (copy-paste, run once):
```
pip install -r requirements.txt
```
If `pip` isn't recognized on Windows, try `python -m pip install -r requirements.txt`.

**4. Train the model** (copy-paste, run once — takes about 1-2 minutes,
since it now trains and compares 4 different algorithms plus tuning and
cross-validation):
```
python train_model.py
```
You'll see clearly labelled progress for every methodology step,
finishing with "All done!" and a final accuracy summary.

**5. Launch the app — choose ONE of the two options below:**

**Option A — Desktop app:**
```
python gui_app.py
```
A window opens. Fill in the dropdowns, click **Predict Diagnosis**. Use
the chart dropdown + "View Chart" button to see Feature Importance, SHAP,
or Model Comparison charts.

**Option B — Web app (also works on phones):**
```
streamlit run web_app.py
```
Opens automatically at `http://localhost:8501`. To use it on your phone
(same Wi-Fi network), check the terminal for a `Network URL` line and
open that address in your phone's browser. All three explainability
charts are available in expandable sections below the prediction.

**Note for Linux users:** if `gui_app.py` gives a `ModuleNotFoundError:
No module named 'tkinter'`, run `sudo apt install python3-tk` first.
(Not needed on Windows/Mac.)

## How the model works (for a viva/demo)

- **Algorithms compared**: Decision Tree, Random Forest, Support Vector
  Machine, and Neural Network (MLP) — all four are trained and evaluated
  on a held-out validation set; the best performer is automatically
  selected, then fine-tuned with hyperparameter search (GridSearchCV).
- **Data split**: 70% training / 15% validation / 15% test — the test set
  is never touched until the very final evaluation, so reported numbers
  are honest, not optimistic.
- **Robustness check**: 5-fold cross-validation confirms the model
  performs consistently across different data splits, not just one lucky
  split.
- **Performance** (see `model_report.txt` for full numbers): accuracy
  ≈ 99%, AUC-ROC ≈ 0.999 on the untouched test set.
- **Explainability — two methods**:
  - *Feature importance* (`feature_importance.png`) — a simple ranking of
    which symptoms the model relies on most.
  - *SHAP* (`shap_summary.png`) — a more advanced, per-prediction
    explanation showing not just which symptoms matter, but whether a
    high or low value of each one pushes the prediction toward or away
    from IDA. This directly matches the proposal's request for SHAP-based
    explainability.
  - Both agree: **Fatigue** and **Paleness** are by far the strongest
    predictors, which matches real-world clinical understanding of IDA —
    a good sanity check that the model learned something medically
    sensible rather than a random pattern.

## Important note

This tool is built for an academic/research project and is **not** a
certified medical device. It should always be explained to viewers as a
screening aid, not a replacement for clinical diagnosis or lab testing.
The "Clinical Validation" and "Deployment" stages described in the
proposal are real-world research steps that would need actual doctors
and hospital infrastructure — they are intentionally outside the scope of
this coded deliverable.

## Doctor View password — IMPORTANT security note

Both `web_app.py` and `gui_app.py` protect Doctor View (charts, model
stats, per-patient explanations) behind a password. The current default
(local-only fallback) password is `YTIdTlU6AMJyUp`.

**⚠️ If your GitHub repository is Public, do NOT rely on the password
written in the code** — anyone can view your source code on GitHub and
read the password directly. For the desktop app (`gui_app.py`) this is
lower risk since it usually isn't shared publicly, but for the deployed
web app you should set a real secret:

**On Streamlit Community Cloud:**
1. Go to your app's page on share.streamlit.io
2. Click the **⋮** menu (top right) → **Settings** → **Secrets**
3. Add this, replacing `yourpassword` with a real password only you and
   the doctors know:
   ```
   DOCTOR_PASSWORD = "yourpassword"
   ```
4. Save. The app will automatically use this instead of the default
   `YTIdTlU6AMJyUp` fallback in the code — and this secret is **never**
   visible in your public GitHub repository.

**Running locally:** create a file `.streamlit/secrets.toml` (this file
should NOT be uploaded to GitHub — add it to `.gitignore`) with the same
`DOCTOR_PASSWORD = "yourpassword"` line.

## Updates per Dr. Ruchita Dixit's research protocol (July 2026)

Two additions were made based on the detailed protocol document she sent:

**1. Clinical evaluation metrics (Sensitivity, Specificity, PPV, NPV, ROC curve)**
`train_model.py` now computes and reports these standard diagnostic-test
metrics alongside the existing Accuracy/Precision/Recall/F1/AUC. They're
saved to `clinical_metrics.txt` and `roc_curve.png` after every training
run.

**2. Modified Kuppuswamy Socioeconomic Scale**
Both apps now automatically calculate the patient's socioeconomic class
(Upper / Upper-Middle / Lower-Middle / Upper-Lower / Lower) from their
Education, Occupation, and Income inputs, using the official scoring
sheet Dr. Dixit provided. This is shown in Doctor View only, alongside
the prediction result.

**Note on the Occupation/Income mapping:** Our app's Occupation and
Income categories were originally built from a different coding
document (used to train the AI model) and don't exactly match the
Kuppuswamy scale's categories one-to-one. The mapping tables in the code
(`KUPPUSWAMY_OCCUPATION_SCORE`, `KUPPUSWAMY_INCOME_SCORE`) are a
reasonable best-effort approximation — Education maps directly and
exactly, but Occupation and Income are approximated to the closest
Kuppuswamy category/bracket. If exact scoring matters for the research
paper, it's worth double-checking these mappings with Dr. Dixit.

**3. Patient Data Collection — now with a real, permanent cloud database**
Both apps have a **separate, separately-password-protected** "Data
Collection" section (distinct from Doctor View) where a doctor can enroll
a real patient — recording demographics, symptoms, lab values
(Hemoglobin, Ferritin, Serum Iron, TIBC, Transferrin Saturation), the
lab-confirmed IDA diagnosis, and the AI model's own prediction for
comparison. Records can be viewed and exported to CSV at any time for
statistical analysis.

**Managing records (hide / restore / permanently delete):** Every
record can be **hidden** (removed from the main list but kept safely in
the database — reversible anytime via a "Restore" option), or
**permanently deleted** (irreversible, requires an explicit confirmation
step first). Hiding is the recommended default for research data — it
keeps the dataset intact for later statistical analysis while still
letting you declutter the working view.

Default data collection password: `gWdthmEiH71SUZ` (change this — see
password setup instructions above; same process as the Doctor View
password, just using the key `DATA_ENTRY_PASSWORD` instead).

**Storage automatically upgrades itself:** if no cloud database is
configured, the app quietly falls back to a local SQLite file so it
still works out of the box for testing. Once you connect a real cloud
database (steps below), it automatically switches to that instead — no
code changes needed, and the app tells you which mode it's in every time
you open Data Collection.

### Setting up the permanent cloud database (Supabase — free)

This only needs to be done **once**. Both `web_app.py` and `gui_app.py`
will then share the same real database, and data survives app restarts,
redeploys, and sleep/wake cycles.

**Step 1 — Create a free Supabase account and project**
1. Go to [supabase.com](https://supabase.com) and sign up (free, no credit card).
2. Click **New Project**. Give it any name (e.g. `ida-screening`), set a
   database password (write it down), choose the region closest to you,
   and click **Create new project**. Wait ~2 minutes for it to provision.

**Step 2 — Get the connection string**
1. In your new project, go to **Project Settings** (gear icon) → **Database**.
2. Under **Connection string**, choose the **URI** tab.
3. Copy the string — it looks like:
   `postgresql://postgres:[YOUR-PASSWORD]@db.xxxxxxxxxxxx.supabase.co:5432/postgres`
4. Replace `[YOUR-PASSWORD]` with the database password you set in Step 1.

**Step 3 — Connect the web app (Streamlit Cloud)**
1. Go to your app on [share.streamlit.io](https://share.streamlit.io).
2. Click **⋮** → **Settings** → **Secrets**.
3. Add this line (alongside your existing `DOCTOR_PASSWORD` line):
   ```
   SUPABASE_DB_URL = "postgresql://postgres:yourpassword@db.xxxxxxxxxxxx.supabase.co:5432/postgres"
   ```
4. Save. The app will automatically reconnect to the cloud database —
   you'll see a green "☁️ Connected to cloud database" message in the
   Data Collection section from then on.

**Step 4 — Connect the desktop app (`gui_app.py`)**
1. In your project folder, create a new file named exactly `db_config.txt`.
2. Paste just the connection string (from Step 2) as the only line in
   that file. Save it.
3. **Do NOT upload this file to GitHub** — it contains your database
   password. It's already excluded via `.gitignore` if you're using Git;
   if not, just make sure to never drag it into a GitHub upload.
4. Run `python gui_app.py` — it will now use the same cloud database as
   the web app.

**That's it.** Both apps now write to the same real, permanent database.
You can also view/query the data directly anytime in Supabase's own
**Table Editor** (in your project dashboard) — it looks and works like a
spreadsheet, no SQL knowledge needed.

**Local testing without any of this:** if you just want to try the Data
Collection feature locally without setting up Supabase yet, do nothing —
it automatically uses a local SQLite file (`patient_records.db`) in the
meantime. You'll see a yellow "no cloud database configured" note
reminding you it's local-only.

## Updates per Dr. Ruchita Dixit's "Required Modifications" email (30 July 2026)

**1. Demographics — updated to match her sheet exactly**
Gender now has **Other** in addition to Male/Female. Education, Occupation,
and Income options were replaced with the exact wording/brackets from her
"Demographic and Socio-Economic Status" PDF.

**2. Signs & Symptoms — updated to match her sheet exactly**
All 23 symptoms now use the exact Hindi/English wording and 4-level
(Absent/Mild/Moderate/Severe) descriptions from her standardized symptoms
document, instead of the shorter labels used before. This makes the form
much easier for rural patients to read and understand.
*Important caveat:* the AI model was trained on the original 10,000+10,000
patient dataset, where several of these symptoms only had 2–3 severity
levels recorded (not the full 0–3 range). The form now **collects** the
full 4-level detail for every symptom (useful for future retraining and
for the doctors' own records), but the AI's prediction for the
rarely-seen levels of those particular symptoms is a bit less certain
until the model is retrained on data that actually includes them. This
doesn't break anything — it's just worth mentioning in a viva if asked.

**3. Automatic Socioeconomic Classification — now exact, not approximate**
Because Education and Income options above now match Dr. Dixit's official
scoring sheet exactly, `calculate_kuppuswamy()` no longer needs to
approximate those two — only **Occupation** still uses a best-effort
mapping (marked `CONFIRM` in the code) because her sheet's categories
(Student, Homemaker, Govt/Private employee, Business, etc.) don't
correspond 1:1 to the official Kuppuswamy occupation categories. Worth a
5-minute check with her before the final paper.

**4/5/6. Laboratory Data Integration — fully automatic now**
- The single-patient entry form no longer asks staff to manually pick
  "IDA Positive/Negative" — it's calculated automatically from the 5 lab
  values using the exact WHO-based cut-offs in her "standard diagnostic
  criteria" PDF (`classify_lab_ida()` in the code).
- A new **"🧪 Import Lab Report"** tab (web app → Data Collection) lets a
  doctor/lab upload the pathology lab's Excel/CSV file directly. The AI
  matches each row to the correct patient by **Patient ID / Registration
  Number**, fills in the lab values, and assigns Laboratory-Confirmed IDA
  Positive/Negative automatically — this is the "connect to the pathology
  lab" workflow she asked about. (Currently built into the web app only;
  the desktop app still takes lab values one patient at a time.)

**No interface or font changes were made** — Dr. Dixit approved the
current look, so only the data/logic described above was touched.

## Update — Training dataset now generated from the full Lab Report, in sync with the apps

Per Mam's explanation of the intended training/testing/deployment
workflow (case-control training with symptoms + lab reports; AI learns
symptom weightage; testing = symptoms-only prediction, cross-checked
against the lab report; final deployment = symptoms-only, no lab needed
in rural areas):

- **`generate_dataset.py` regenerated.** Every synthetic patient now gets
  the full 29-parameter Lab Report (CBC + Peripheral Smear + Iron
  Profile), not just the old 5 values. `Outcome` (case/control) is
  computed with the exact same `classify_lab_verdict()` core-panel rule
  (`core_met == 5`, i.e. Hb + Ferritin + Serum Iron + TIBC + TSAT all
  meeting cut-off) that `web_app.py`/`gui_app.py` already use to confirm
  IDA from a real patient's lab report — so the label the model trains
  on can never drift out of sync with what the app tells a doctor.
  Non-core CBC/PS/iron-profile findings are correlated with the outcome
  probabilistically (not forced), so they carry a genuine but non-trivial
  signal, same as the symptom sampling.
- **`train_model.py` unchanged in approach** — it already excludes all
  lab-value columns from the model's input features (only demographics +
  symptoms go in), it just now excludes all 33 new `cbc_/ps_/iron_`
  columns instead of the old 5. This is exactly the "AI predicts from
  symptoms alone" behaviour Mam described for testing/deployment.
- **Retrained model:** 99.7% test accuracy, AUC 0.9999 on the new
  20,000-record dataset (SVM selected as best model).

**Worth confirming with Dr. Dixit:** the `core_met == 5` boundary (all 5
of Hb/Ferritin/Iron/TIBC/TSAT must meet cut-off) is what generates the
case/control labels. This mirrors the exact rule already coded into
`classify_lab_verdict()`, but is a stricter, more "textbook" definition
than some real-world case/control datasets use (some allow 4/5 markers).
If she'd prefer a looser matching rule, that's a one-line change in
`classify_lab_verdict()` (both apps) plus a re-run of `generate_dataset.py`
+ `train_model.py`.

## Update — Removed 4 expensive/advanced Iron Profile markers

Per Dr. Dixit's feedback: sTfR, Ret-He/CHr, CRP, and Hepcidin need
specialised equipment/reagents that most labs (especially in
rural/resource-limited settings) don't have — not every patient could
realistically get these done, which worked against the project's goal
of a low-cost, widely usable screening tool.

Removed from `LAB_IRON_FIELDS` in `web_app.py`, `gui_app.py`, and
`generate_dataset.py`. The Iron Profile panel is now the 5 standard,
widely-available tests: **Serum Ferritin, Serum Iron, TIBC,
Transferrin, TSAT** — these are also exactly the 5 `LAB_CORE_PARAMS`
that already drove the IDA verdict, so the core diagnostic logic is
unaffected. Total Lab Report panel is now 29 parameters (9 CBC + 15 PS
+ 5 Iron Profile), down from 33.

`generate_dataset.py` was re-run (20,000 records, full 29-parameter
panel) and `train_model.py` retrained on the new dataset — model
inputs (symptoms + demographics only) and performance are unaffected
by this change, since these 4 markers were never model features to
begin with (only used for supportive notes in the lab verdict).

## Update — Iron Profile removed entirely; IDA verdict now calculated from CBC + Peripheral Smear only

Per further instruction: the Iron Profile panel (Serum Ferritin, Serum
Iron, TIBC, Transferrin, TSAT) has been removed **completely** — it is
no longer collected in either app, no longer stored in the database, and
no longer used anywhere in the lab-verdict calculation. Nothing is
calculated from it any more.

**The core diagnostic panel is now CBC-based**, not iron-based. The 5
CORE parameters that drive the "Suggestive / Not suggestive of IDA"
verdict in `classify_lab_verdict()` (`web_app.py`, `gui_app.py`,
`generate_dataset.py`) are now:

| Old core panel (removed) | New core panel |
|---|---|
| Hb, Ferritin, Serum Iron, TIBC, TSAT | **Hb, MCV, MCH, MCHC, RDW-CV** |

This is the classic microcytic-hypochromic CBC picture of IDA. Peripheral
Smear findings (RBC size, staining, anisocytosis, poikilocytosis, pencil
cells, etc.) remain supportive evidence, exactly as before — only the
Iron Profile is gone. The Lab Report panel is now **24 parameters (9 CBC
+ 15 Peripheral Smear)**, down from 29.

**What changed, file by file:**
- `web_app.py` / `gui_app.py` — Iron Profile input section removed from
  both the "Add New Patient" form and the "Update Lab Report" form; the
  database's quick-access lab columns (`serum_ferritin`, `serum_iron`,
  `tibc`, `transferrin_saturation`) were replaced with `mcv`, `mch`,
  `mchc`, `rdwcv` (with a safe migration for existing databases — old
  Iron Profile columns are simply left unused, nothing is deleted).
- `generate_dataset.py` — no longer generates or saves any `iron_*`
  columns; synthetic patients' case/control label (`Outcome`) is now
  determined purely from the CBC core panel above.
- `train_model.py` — unaffected in approach (it already excludes all
  `cbc_`/`ps_` lab columns from the model's inputs; there's simply
  nothing to exclude from an Iron Profile any more, since it doesn't
  exist in the dataset).
- **Dataset regenerated and model retrained** on the new 20,000-record,
  24-lab-column dataset: **99.47% test accuracy, AUC 0.9998** (SVM
  selected as best model) — consistent with before, since the model's
  own inputs (symptoms + demographics) never changed.

**Worth confirming with Dr. Dixit:** the new CBC core panel (Hb, MCV,
MCH, MCHC, RDW-CV all meeting cut-off) is a reasonable, textbook
microcytic-hypochromic definition, but it's a different clinical
boundary than the WHO iron-studies definition used before. If she'd
prefer a different combination of CBC parameters for the core panel,
that's a one-line change to `LAB_CORE_PARAMS` (all 3 files) plus a
re-run of `generate_dataset.py` + `train_model.py`.

## Update — Peripheral Smear given equal weight in the core panel

Per further instruction: Peripheral Smear findings are no longer purely
supportive — 5 of the 15 PS findings most specific to IDA are now part
of the **CORE** panel, given equal weight to the 5 CBC indices.

**New 10-parameter CORE panel** (`LAB_CORE_PARAMS` in `web_app.py`,
`gui_app.py`, `generate_dataset.py`):

| CBC core (5) | Peripheral Smear core (5) |
|---|---|
| Hb | RBC size = Microcytic |
| MCV | RBC staining = Hypochromic |
| MCH | Anisocytosis = Present |
| MCHC | Poikilocytosis = Present |
| RDW-CV | Pencil cells / Elliptocytes = Present |

`classify_lab_verdict()`'s boundary is now `core_met == 10` for a clean
"SUGGESTIVE of IDA" verdict, `core_met == 0` for "NOT suggestive", and
anything from 1–9 is "Equivocal — only X/10 core parameters meet the IDA
cut-off; correlate clinically". The remaining CBC indices (RBC count,
Hematocrit) and the other 10 PS findings (Target cells, Teardrop cells,
Schistocytes, Rouleaux, Polychromasia, Nucleated RBCs, WBC/Platelet
morphology, Platelet number, Hemoparasites — these point more toward
other conditions like thalassemia, hemolysis, or myelofibrosis than IDA
specifically) remain SUPPORTIVE-only, exactly as before.

`generate_dataset.py`'s synthetic-patient generation was updated to
match: for IDA-negative patients, the "forced normal" parameters are now
drawn from the full 10-parameter core (CBC + PS combined) rather than
just the 5 CBC ones, so the dataset's `Outcome` label stays in sync with
the new `core_met == 10` boundary.

**Dataset regenerated and model retrained** on the new 20,000-record
dataset (same 24 lab columns, just a different core/supportive split):
**99.47% test accuracy, AUC 0.9997** — again unaffected by this change,
since the model's inputs are still symptoms + demographics only.

**Worth confirming with Dr. Dixit:** which 5 of the 15 PS findings
should count as "core" (this pick — RBC size, RBC staining,
Anisocytosis, Poikilocytosis, Pencil cells — was chosen because they're
the textbook IDA-specific findings; the other 10 are more specific to
other blood disorders). If she'd prefer a different set, or a different
number of PS findings in the core, that's a one-line change to
`CORE_PS_PARAMS` (all 3 files) plus a re-run of `generate_dataset.py` +
`train_model.py`.

## Update — Simplified lab verdict rule (no more "X/5" / "X/10" scoring), per Dr. Dixit

The 10-parameter "core count" scoring system has been replaced entirely
with a simple, deterministic rule Dr. Dixit specified directly — no more
fractional "X/5" or "X/10" language anywhere in the verdict.

**CBC criterion** (gender-specific microcytic-hypochromic pattern):
- Female: Hb < 12 g/dL **AND** MCV < 80 fL **AND** MCH < 27 pg
- Male: Hb < 13 g/dL **AND** MCV < 80 fL **AND** MCH < 27 pg
- (Gender = Other, or not provided: uses the female/lower 12 g/dL cutoff
  as the safer, more inclusive default — confirm with Dr. Dixit if a
  different default is preferred)

**PS criterion:**
- RBC size = **Microcytic** **AND** RBC staining = **Hypochromic**
- Per Dr. Dixit and this hospital's reporting convention, Anisocytosis,
  Poikilocytosis and Pencil cells are only written up in the report when
  present — so Microcytic + Hypochromic reliably implies the full classic
  IDA smear picture (Anisocytosis/Poikilocytosis/Pencil cells present, WBC
  morphology normal, Platelets adequate/normal, Hemoparasites not seen),
  without needing to check those separately.

**Final verdict: CBC criterion met OR PS criterion met** → "Laboratory
findings SUGGESTIVE of Iron Deficiency Anaemia". Either one alone is
sufficient — they are independent diagnostic pathways to the same
conclusion, not both required together. If both are entered and neither
is met → "NOT suggestive". If only one panel (CBC or PS) has been entered
and it doesn't meet its criterion, the verdict says which panel is still
needed rather than guessing.

**Changed everywhere the verdict is calculated:** `classify_lab_verdict()`
in `web_app.py`, `gui_app.py`, `generate_dataset.py`, and
`recalculate_lab_verdicts.py` — all four now take an optional `gender`
parameter (1=Male, 2=Female, 3=Other) needed for the CBC criterion's
gender-specific Hb cutoff. `generate_dataset.py`'s synthetic-patient
generation was rewritten to match: positive patients are randomly
assigned to satisfy the CBC pathway only, PS pathway only, or both (45%
/ 45% / 10%), so the training data reflects that either pathway alone is
diagnostic — Outcome = 1 iff `cbc_met OR ps_met`.

**Dataset regenerated and model retrained** on the new 20,000-record
dataset: **99.67% test accuracy, AUC 0.9999** — again unaffected in
approach, since the model's own inputs are still symptoms + demographics
only; only the label (Outcome) generation logic changed.

**PS now shown in the "Laboratory & AI Verdict" section — renamed to
"Laboratory Report (CBC + Peripheral Smear) & Verdict".** All CBC and PS
values are shown together (with "Not entered" for any gaps), addressing
Dr. Dixit's note that PS findings were not appearing alongside CBC in
that detail view.

## Update — Peripheral Smear gaps now visible; AI prediction hidden from live views (for now)

Two fixes based on field feedback after the first ~34 patients were entered:

**1. "Not entered" now shown for missing lab fields, instead of the field
disappearing.** In the "View / Export Records" → "View Full Record" detail
panel, the lab section used to silently hide any field with no value. This
meant that if an intern skipped a Peripheral Smear dropdown (they default to
"Not selected" — easy to miss, unlike the CBC number fields which always
carry *some* value), the review screen looked like PS was never part of the
form at all, instead of showing it as a visible gap. The section (renamed
"🧪 Laboratory Report (CBC + Peripheral Smear) & Verdict") now shows every
CBC + PS field for every patient, with **"Not entered"** displayed for
anything missing — so gaps are obvious at a glance and can be filled in via
"Update Lab Report" if the original report/slide is still available.

**2. AI prediction (`ai_predicted_ida` / `ai_confidence`) hidden from live,
patient-facing/data-collection views — per Dr. Dixit's instruction.** The
model is trained on synthetic (computer-generated) data only and hasn't
been validated on real patients yet, so showing its live +/- output next to
lab-confirmed results during data collection risked confusing interns when
the two disagreed (which is expected — see the "AI trains on lab-confirmed
labels but predicts from symptoms alone" discussion elsewhere in this doc,
this is not a bug). Specifically removed from:
  - the confirmation message shown right after saving a new patient
    (`web_app.py` / `gui_app.py`)
  - the compact records overview table and the GUI's record list
  - the "Laboratory & AI Verdict" detail panel (now "Laboratory Report...")

**Nothing about the AI model or training pipeline was removed** —
`ai_predicted_ida` / `ai_confidence` are still computed and saved to the
database and Excel export for every patient, exactly as before. This is a
display-only change and is easily reversed (or made permanent) once the
team decides how/when to reintroduce AI predictions after real-patient
validation. If a full removal (delete the model/training code entirely) is
preferred instead, that's a separate, larger change — ask and it can be
done.

## Update — recalculate_lab_verdicts.py (keep old + new patients on the same rule)

Since `LAB_CORE_PARAMS` changed more than once (Iron Profile → CBC-only
core → CBC+PS 10-parameter core), patients enrolled before a given code
update have a `lab_confirmed_ida` verdict computed with whatever rule was
live at the time — it does **not** automatically update just because the
code changes. New patients enrolled after a code update get the current
rule.

`recalculate_lab_verdicts.py` fixes this without needing to re-test
anyone: every record already stores its full raw lab panel in
`lab_panel_json` (this never changes), so the script re-reads that raw
data for every patient and recalculates `lab_confirmed_ida` using
whatever `classify_lab_verdict()` logic is current — bringing every
record (old and new) onto the exact same rule.

```
python recalculate_lab_verdicts.py            # dry run — shows what would change, writes nothing
python recalculate_lab_verdicts.py --apply     # writes the updates (asks for a YES confirmation first)
```

It connects to the database the same way the app does (`SUPABASE_DB_URL`
from environment or `.streamlit/secrets.toml`, falling back to the local
`patient_records.db` SQLite file). Run this once after any future change
to `LAB_CORE_PARAMS` / `classify_lab_verdict()`, so the dataset stays
consistent for analysis and publication.

## Pre-deployment fixes (before real-patient rollout, 1-2 Sept)

- **Fixed a crash bug in web_app.py**: entering a Patient ID that already
  exists (e.g. a returning patient / follow-up visit) used to throw a
  `NameError` and crash the Data Collection page, because the duplicate-ID
  confirmation box referenced `kup_score`/`kup_class` before they were
  computed. These are now computed once up front and passed through.
- **Logo added to the desktop app** (`gui_app.py`) — window icon +
  header, matching the web app's existing droplet/pulse logo.
- **Confirmed already in place** (no change needed): patient/guardian
  consent checkbox (required before saving), duplicate Patient ID
  detection with confirm-to-override, Excel/CSV record export in both
  apps, and `min=0` validation on every lab numeric field.
- **`.streamlit/secrets.toml` is excluded from every project zip** going
  forward — only a `.example` version with placeholders ships. The live
  Supabase DB password must still be rotated manually from the Supabase
  dashboard (Project Settings → Database → Reset database password) —
  this can't be done from code.
- **Reminder:** the AI model is trained entirely on synthetic data. Real
  clinical accuracy is unverified — follow Mam's testing-phase plan
  (predict from symptoms, then cross-check against real lab results)
  before trusting the model's output for real patients.
