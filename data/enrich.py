import pandas as pd
import numpy as np

# -------------------------------
# LOAD + CLEAN
# -------------------------------
df = pd.read_csv("./data/dataset_PFE_CDM_complet.csv")

# Clean column names (CRITICAL)
df.columns = df.columns.str.strip()

# Parse datetime
df["Time"] = pd.to_datetime(df["Time"])

# Ensure numeric
df["Transaction Amount"] = pd.to_numeric(df["Transaction Amount"], errors="coerce")

# -------------------------------
# STEP 1: ACCOUNT-LEVEL AGGREGATION
# -------------------------------
agg = df.groupby("Account Number").agg({
    "Time": ["min", "max", "count"],
    "Transaction Amount": ["mean", "std", "max", "sum"],
    "Country": lambda x: (x != "MA").mean(),
    "Channel": lambda x: (x == "WEB").mean()
})

agg.columns = [
    "first_txn", "last_txn", "txn_count",
    "avg_amount", "std_amount", "max_amount", "total_amount",
    "intl_ratio", "web_ratio"
]

agg = agg.reset_index()

# -------------------------------
# ADD STATIC CLIENT INFO (IMPORTANT FIX)
# -------------------------------
client_info = df.groupby("Account Number").agg({
    "Age": "first",
    "Gender": "first",
    "City": "first",   # ✅ correct client city
    "Bank": "first"
}).reset_index()

agg = agg.merge(client_info, on="Account Number", how="left")

# -------------------------------
# STEP 2: CLIENT TYPE (PERCENTILE-BASED)
# -------------------------------
p90_total = agg["total_amount"].quantile(0.90)
p75_total = agg["total_amount"].quantile(0.75)

p90_max = agg["max_amount"].quantile(0.90)
p75_max = agg["max_amount"].quantile(0.75)

p90_txn = agg["txn_count"].quantile(0.90)
p75_txn = agg["txn_count"].quantile(0.75)

def assign_client_type(row):
    score = 0

    if row["total_amount"] > p90_total:
        score += 2
    elif row["total_amount"] > p75_total:
        score += 1

    if row["max_amount"] > p90_max:
        score += 2
    elif row["max_amount"] > p75_max:
        score += 1

    if row["txn_count"] > p90_txn:
        score += 2
    elif row["txn_count"] > p75_txn:
        score += 1

    if row["intl_ratio"] > 0.2:
        score += 1

    if score >= 5:
        return "PMO"
    elif score >= 2:
        return "PRO"
    else:
        return "PPH"

agg["Client Type"] = agg.apply(assign_client_type, axis=1)

# -------------------------------
# STEP 3: ACCOUNT TYPE
# -------------------------------
def assign_account_type(client_type):
    if client_type == "PPH":
        return np.random.choice(["Checking", "Savings"], p=[0.7, 0.3])
    elif client_type == "PRO":
        return "Business"
    else:
        return "Corporate"

agg["Account Type"] = agg["Client Type"].apply(assign_account_type)

# -------------------------------
# STEP 4: ACCOUNT CREATION DATE
# -------------------------------
def generate_creation_date(row):
    first_txn = row["first_txn"]
    age = row["Age"]

    if age < 30:
        days_back = np.random.randint(30, 365 * 3)
    elif age < 50:
        days_back = np.random.randint(365, 365 * 8)
    else:
        days_back = np.random.randint(365 * 2, 365 * 15)

    if row["txn_count"] > 300:
        days_back += np.random.randint(365, 365 * 5)

    return first_txn - pd.Timedelta(days=int(days_back))

agg["Account Creation Date"] = agg.apply(generate_creation_date, axis=1)

# -------------------------------
# STEP 5: MULTI-ACCOUNT CLIENTS (FIXED)
# -------------------------------
client_ids = {}
current_client_id = 0

for _, row in agg.sample(frac=1).iterrows():
    acc = row["Account Number"]

    if acc in client_ids:
        continue

    # probability based on client type
    if row["Client Type"] == "PMO":
        probs = [0.4, 0.4, 0.2]
    elif row["Client Type"] == "PRO":
        probs = [0.6, 0.3, 0.1]
    else:
        probs = [0.85, 0.14, 0.01]

    n_accounts = np.random.choice([1, 2, 3], p=probs)

    # ✅ FIX: use CLIENT CITY (not City 2)
    candidates = agg[
        (agg["Client Type"] == row["Client Type"]) &
        (agg["City"] == row["City"]) &
        (agg["Bank"] == row["Bank"])
    ]["Account Number"].values

    np.random.shuffle(candidates)

    grouped = 0
    for c in candidates:
        if c not in client_ids:
            client_ids[c] = current_client_id
            grouped += 1
        if grouped >= n_accounts:
            break

    current_client_id += 1

agg["Client ID"] = agg["Account Number"].map(client_ids)

# -------------------------------
# STEP 6: FINAL FEATURES
# -------------------------------
accounts_per_client = agg.groupby("Client ID")["Account Number"].count()

agg["Accounts per Client"] = agg["Client ID"].map(accounts_per_client)
agg["Has Multiple Accounts"] = agg["Accounts per Client"] > 1

# -------------------------------
# MERGE BACK
# -------------------------------
df = df.merge(
    agg[[
        "Account Number",
        "Client Type",
        "Account Type",
        "Account Creation Date",
        "Client ID",
        "Accounts per Client",
        "Has Multiple Accounts"
    ]],
    on="Account Number",
    how="left"
)

# -------------------------------
# FINAL CLEAN
# -------------------------------
df = df.sort_values(["Account Number", "Time"])

# -------------------------------
# SAVE
# -------------------------------
df.to_csv("./data/dataset.csv", index=False)