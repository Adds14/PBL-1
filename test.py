# predict.py
# Loads the saved model and runs interactive prediction.
# Usage: python predict.py
# Requires: model_artifacts_maximal.pkl (generated from FINAL_MODEL_F.ipynb)

import os, io, re
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

ma = joblib.load("model_artifacts_optimal.pkl")

lr_model       = ma["lr_model"]
xgb_calibrated = ma["xgb_calibrated"]
scaler         = ma["scaler"]
pt             = ma["pt"]
train_columns  = ma["train_columns"]
best_w_lr      = ma["best_w_lr"]
best_w_xgb     = ma["best_w_xgb"]
best_thresh    = ma["best_thresh"]

# ─────────────────────────────────────────────
# VALID OPTIONS
# ─────────────────────────────────────────────

VALID_JOBS      = ['admin.','blue-collar','entrepreneur','housemaid','management',
                   'retired','self-employed','services','student','technician',
                   'unemployed','unknown']
VALID_MARITAL   = ['divorced','married','single']
VALID_EDUCATION = ['primary','secondary','tertiary','unknown']
VALID_CONTACT   = ['cellular','telephone','unknown']
VALID_MONTHS    = ['jan','feb','mar','apr','may','jun',
                   'jul','aug','sep','oct','nov','dec']
VALID_POUTCOME  = ['failure','other','success','unknown']
REQUIRED_COLS   = ['age','job','marital','education','default','balance',
                   'housing','loan','contact','day','month','duration',
                   'campaign','pdays','previous','poutcome']

# Categorical columns and their drop_first reference category
# (must match exactly how get_dummies(drop_first=True) worked during training)
CAT_COLS = {
    'job':       VALID_JOBS,       # drop_first drops 'admin.'
    'marital':   VALID_MARITAL,    # drop_first drops 'divorced'
    'education': VALID_EDUCATION,  # drop_first drops 'primary'
    'contact':   VALID_CONTACT,    # drop_first drops 'cellular'
    'month':     VALID_MONTHS,     # drop_first drops 'apr' (first alphabetically after sort)
    'age_group': ['Young','Adult','Senior','Old'],  # drop_first drops 'Young'
    'pdays_bucket': ['recent','warm','cold','never'],  # drop_first drops 'recent'
}

# ─────────────────────────────────────────────
# INPUT HELPERS
# ─────────────────────────────────────────────

def ask_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val is not None and val < min_val:
                print(f"  Must be >= {min_val}.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Must be <= {max_val}.")
                continue
            return val
        except ValueError:
            print("  Enter a whole number.")

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Enter a number.")

def ask_choice(prompt, options):
    while True:
        val = input(f"  {prompt} [{' / '.join(options)}]: ").strip().lower()
        if val in options:
            return val
        print(f"  Invalid. Choose from: {', '.join(options)}")

def ask_yesno(prompt):
    while True:
        val = input(f"  {prompt} [yes / no]: ").strip().lower()
        if val in ('yes', 'no'):
            return val
        print("  Enter 'yes' or 'no'.")


def collect_customer(n):
    print(f"\n{'─'*50}")
    print(f"  Customer {n}")
    print(f"{'─'*50}")

    name = input("  Name: ").strip() or f"Customer {n}"

    print("\n  [ Personal ]")
    age       = ask_int("  Age: ", 18, 95)
    job       = ask_choice("Job", VALID_JOBS)
    marital   = ask_choice("Marital status", VALID_MARITAL)
    education = ask_choice("Education", VALID_EDUCATION)
    default   = ask_yesno("Credit in default?")

    print("\n  [ Financial ]")
    balance = ask_float("  Account balance in € (can be negative): ")
    housing = ask_yesno("Has housing loan?")
    loan    = ask_yesno("Has personal loan?")

    print("\n  [ Last Campaign Contact ]")
    contact  = ask_choice("Contact type", VALID_CONTACT)
    day      = ask_int("  Day of month contacted: ", 1, 31)
    month    = ask_choice("Month contacted", VALID_MONTHS)
    duration = ask_int("  Call duration in seconds: ", 0)
    campaign = ask_int("  Number of contacts this campaign: ", 1)

    print("\n  [ Previous Campaign ]")
    pdays    = ask_int("  Days since last contact (-1 = never): ", -1)
    previous = ask_int("  Number of contacts before this campaign: ", 0)
    poutcome = ask_choice("Outcome of previous campaign", VALID_POUTCOME)

    return {
        "name": name, "age": age, "job": job, "marital": marital,
        "education": education, "default": default, "balance": balance,
        "housing": housing, "loan": loan, "contact": contact, "day": day,
        "month": month, "duration": duration, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome,
    }


# ─────────────────────────────────────────────
# CSV INPUT
# ─────────────────────────────────────────────

def load_from_csv(path):
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None

    try:
        with open(path, 'r') as f:
            raw = f.read()

        fixed = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = re.sub(r'""([^";]*)""', r'\1', line)
            line = line.replace('"', '')
            fixed.append(line)

        df = pd.read_csv(io.StringIO('\n'.join(fixed)), sep=';')
        df.columns = df.columns.str.strip()

    except Exception as e:
        print(f"  Error reading file: {e}")
        return None

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  CSV missing columns: {missing}")
        print(f"  Columns found: {list(df.columns)}")
        return None

    if 'name' not in df.columns:
        df['name'] = [f"Customer {i+1}" for i in range(len(df))]

    print(f"  Loaded {len(df)} customers from {path}")
    return df[['name'] + REQUIRED_COLS].to_dict(orient='records')


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def manual_dummies(df, col, all_categories):
    """
    Manually one-hot encode a column with drop_first=True.
    This avoids pd.get_dummies missing categories not present in small batches.
    The first category (alphabetically, as pandas sorts) is the dropped reference.
    """
    sorted_cats = sorted(all_categories)
    ref = sorted_cats[0]  # dropped reference category
    for cat in sorted_cats[1:]:
        col_name = f"{col}_{cat}"
        df[col_name] = (df[col] == cat).astype(int)
    df.drop(columns=[col], inplace=True)
    return df


def preprocess(raw_list):
    df = pd.DataFrame(raw_list).drop(columns=['name'], errors='ignore')

    # Drop target column if loaded from CSV
    df.drop(columns=['y'], errors='ignore', inplace=True)

    # Binary encode — handles both object and pandas 2.x StringDtype
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    # Feature engineering (DATA_PREPROCESSING.ipynb)
    df['previous_contacted'] = df['poutcome'].apply(lambda x: 0 if x == 'unknown' else 1)
    df['previous_success']   = df['poutcome'].apply(lambda x: 1 if x == 'success' else 0)
    df['age_group']          = pd.cut(df['age'], bins=[18, 30, 45, 60, 100],
                                      labels=['Young', 'Adult', 'Senior', 'Old'])
    df['age_group']          = df['age_group'].astype(str)
    df.drop('poutcome', axis=1, inplace=True)

    # Power transform
    df[['balance', 'duration']] = pt.transform(df[['balance', 'duration']])

    # Advanced features (AdvancedFeatures.ipynb)
    df['total_contacts']         = df['campaign'] + df['previous']
    df['contact_pressure']       = df['campaign'] / (df['previous'] + 1)
    df['high_contact_flag']      = (df['campaign'] > 3).astype(int)
    df['never_contacted_before'] = (df['pdays'] == -1).astype(int)
    df['recent_contact']         = df['pdays'].apply(lambda x: 1 if 0 <= x <= 30 else 0)
    df['pdays_bucket']           = pd.cut(df['pdays'].replace(-1, 999),
                                          bins=[-1, 30, 90, 180, 999],
                                          labels=['recent', 'warm', 'cold', 'never'])
    df['pdays_bucket']           = df['pdays_bucket'].astype(str)
    df['prev_success']           = (df['previous_success'] == 1).astype(int)
    df['prev_contact_flag']      = (df['previous'] > 0).astype(int)
    df['engagement_score']       = df['previous'] * df['prev_success']

    # Manual one-hot encoding — ensures all training categories are present
    for col, categories in CAT_COLS.items():
        df = manual_dummies(df, col, categories)

    # Align to exact training columns
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[train_columns]

    # Debug: find any non-numeric columns
    for col in df.columns:
        if df[col].dtype == object:
            print(f"  [DEBUG] Non-numeric column still present: {col} → {df[col].tolist()}")

    df = df.astype(float)
    return df, scaler.transform(df)


# ─────────────────────────────────────────────
# PREDICT & RANK
# ─────────────────────────────────────────────

def subscription_tier(p):
    if p >= 0.8: return "Very Likely"
    if p >= 0.6: return "Likely"
    if p >= 0.4: return "Moderate"
    if p >= 0.2: return "Unlikely"
    return "Very Unlikely"


def predict_and_rank(raw_list):
    X_raw, X_scaled = preprocess(raw_list)

    lr_prob  = lr_model.predict_proba(X_scaled)[:, 1]
    xgb_prob = xgb_calibrated.predict_proba(X_raw)[:, 1]

    final_prob = best_w_lr * lr_prob + best_w_xgb * xgb_prob
    final_pred = (final_prob >= best_thresh).astype(int)

    results = pd.DataFrame({
        "Name":          [r["name"]     for r in raw_list],
        "Age":           [r["age"]      for r in raw_list],
        "Job":           [r["job"]      for r in raw_list],
        "Duration(s)":   [r["duration"] for r in raw_list],
        "LR_Prob":       lr_prob.round(4),
        "XGB_Prob":      xgb_prob.round(4),
        "Final_Prob":    final_prob.round(4),
        "Probability_%": (final_prob * 100).round(1),
        "Prediction":    ["Subscribe" if p == 1 else "No Subscribe" for p in final_pred],
        "Tier":          [subscription_tier(p) for p in final_prob],
    })

    results = results.sort_values("Final_Prob", ascending=False).reset_index(drop=True)
    results.index += 1
    results.index.name = "Rank"
    return results


def print_results(results):
    n = len(results)
    print("\n" + "="*70)
    if n == 1:
        print("                     PREDICTION RESULT")
    else:
        print("           RANKED RESULTS  (highest → lowest probability)")
    print("="*70)
    print(results[["Name", "Age", "Job", "Duration(s)",
                   "Probability_%", "Prediction", "Tier"]].to_string())
    print("="*70)
    print(f"\nThreshold: {best_thresh}  |  LR: {best_w_lr}  |  XGB: {best_w_xgb}")
    if n > 1:
        top = results.iloc[0]
        print(f"\nTop target : {top['Name']}  —  {top['Probability_%']}%  ({top['Tier']})")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("    BANK SUBSCRIPTION PREDICTION TOOL")
    print("="*50)
    print(f"  Threshold: {best_thresh}  |  LR: {best_w_lr}  |  XGB: {best_w_xgb}")

    print("\nHow do you want to enter customer data?")
    print("  1. Manual input")
    print("  2. Load from CSV file")

    while True:
        choice = input("\nEnter 1 or 2: ").strip()
        if choice in ('1', '2'):
            break
        print("  Enter 1 or 2.")

    if choice == '1':
        n = ask_int("\nHow many customers to predict? ", min_val=1)
        customers_input = [collect_customer(i + 1) for i in range(n)]
    else:
        while True:
            path = input("\nEnter CSV file path: ").strip().strip('"')
            customers_input = load_from_csv(path)
            if customers_input:
                break
            print("  Please try again.")

    print("\nRunning predictions...")
    results = predict_and_rank(customers_input)
    print_results(results)