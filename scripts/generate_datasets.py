"""
Synthetic dataset generator for the Data Cleaning Agent OpenEnv.

Generates dirty/clean CSV pairs for Easy (Customer, 50 rows),
Medium (Sales, 200 rows), and Hard (HR, 500 rows) difficulty levels.

Usage:
    python scripts/generate_datasets.py
"""

import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# Easy dataset — Customer, 50 rows
# ---------------------------------------------------------------------------

def generate_easy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dirty_df, clean_df) for the Easy Customer dataset."""
    rng = np.random.default_rng(SEED)

    n_clean = 45  # 45 unique rows; 5 duplicates added later → 50 dirty rows

    first_names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank",
        "Iris", "Jack", "Karen", "Leo", "Mia", "Nate", "Olivia", "Paul",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
        "Yara", "Zoe", "Aaron", "Beth", "Carl", "Diana", "Ethan", "Fiona",
        "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Mike",
        "Nancy", "Oscar", "Pam", "Roy", "Sara", "Tom",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
        "White", "Harris", "Martin", "Thompson", "Young", "Allen",
    ]
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    ]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "example.com"]

    # Build clean rows
    rows = []
    for i in range(n_clean):
        fn = first_names[i % len(first_names)]
        ln = last_names[i % len(last_names)]
        name = f"{fn} {ln}"
        cid = 1001 + i
        email = f"{fn.lower()}.{ln.lower()}{cid}@{domains[i % len(domains)]}"
        age = int(rng.integers(18, 70))
        phone = f"555-{rng.integers(100, 999):03d}-{rng.integers(1000, 9999):04d}"
        city = cities[i % len(cities)]
        year = int(rng.integers(2018, 2024))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        signup_date = f"{year}-{month:02d}-{day:02d}"
        rows.append({
            "customer_id": cid,
            "name": name,
            "email": email,
            "age": age,
            "phone": phone,
            "city": city,
            "signup_date": signup_date,
        })

    clean_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Build dirty version from clean
    # ------------------------------------------------------------------
    dirty_df = clean_df.copy()

    # 1. Duplicate rows — repeat rows 0, 5, 10, 15, 20 (5 duplicates)
    dup_indices = [0, 5, 10, 15, 20]
    duplicates = clean_df.iloc[dup_indices].copy()
    dirty_df = pd.concat([dirty_df, duplicates], ignore_index=True)

    # 2. Missing values — scatter NaN across several columns
    missing_positions = [
        ("name", [2, 8, 33]),
        ("email", [1, 14, 27]),
        ("age", [3, 19, 40]),
        ("city", [6, 22, 38]),
    ]
    for col, idxs in missing_positions:
        for idx in idxs:
            dirty_df.at[idx, col] = np.nan

    # 3. Type errors — store age/customer_id as string with non-numeric chars
    dirty_df["age"] = dirty_df["age"].astype(object)
    dirty_df.at[4, "age"] = "abc"          # non-numeric string
    dirty_df.at[11, "age"] = "twenty"      # non-numeric string
    dirty_df.at[25, "age"] = "N/A"         # non-numeric string

    dirty_df["customer_id"] = dirty_df["customer_id"].astype(object)
    dirty_df.at[7, "customer_id"] = "ID#1008"   # string with prefix
    dirty_df.at[17, "customer_id"] = "???"       # garbage

    # Phone numbers with letters (type error)
    dirty_df.at[9, "phone"] = "555-abc-1234"
    dirty_df.at[30, "phone"] = "555-XYZ-5678"

    # 4. Formatting inconsistencies — extra whitespace in name column
    whitespace_idxs = [0, 12, 23, 35, 44]
    for idx in whitespace_idxs:
        val = dirty_df.at[idx, "name"]
        if pd.notna(val):
            dirty_df.at[idx, "name"] = f"  {val}  "

    return dirty_df, clean_df


# ---------------------------------------------------------------------------
# Medium dataset — Sales, 200 rows  (stub — implemented in task 2.2)
# ---------------------------------------------------------------------------

def generate_medium() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dirty_df, clean_df) for the Medium Sales dataset."""
    rng = np.random.default_rng(SEED + 1)

    n_clean = 190  # 190 unique rows; 10 duplicates added later

    products = [
        "Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
        "Webcam", "Desk", "Chair", "Notebook", "Pen",
    ]
    regions = ["North", "South", "East", "West", "Central"]
    reps = [
        "Alice Smith", "Bob Jones", "Carol White", "David Brown",
        "Eve Davis", "Frank Wilson", "Grace Moore", "Hank Taylor",
        "Iris Anderson", "Jack Thomas",
    ]

    rows = []
    for i in range(n_clean):
        sale_id = 2001 + i
        product = products[i % len(products)]
        quantity = int(rng.integers(1, 50))
        unit_price = round(float(rng.uniform(5.0, 500.0)), 2)
        total = round(quantity * unit_price, 2)
        region = regions[i % len(regions)]
        rep = reps[i % len(reps)]
        year = int(rng.integers(2020, 2024))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        sale_date = f"{year}-{month:02d}-{day:02d}"
        rows.append({
            "sale_id": sale_id,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "total": total,
            "region": region,
            "sales_rep": rep,
            "sale_date": sale_date,
        })

    clean_df = pd.DataFrame(rows)
    dirty_df = clean_df.copy()

    # Duplicates
    dup_indices = list(range(0, 50, 5))[:10]
    duplicates = clean_df.iloc[dup_indices].copy()
    dirty_df = pd.concat([dirty_df, duplicates], ignore_index=True)

    # Missing values
    for col, idxs in [
        ("product", [3, 15, 60, 120]),
        ("quantity", [7, 30, 90]),
        ("region", [12, 45, 100, 155]),
        ("sales_rep", [20, 75, 140]),
    ]:
        for idx in idxs:
            dirty_df.at[idx, col] = np.nan

    # Type errors
    dirty_df["quantity"] = dirty_df["quantity"].astype(object)
    dirty_df.at[5, "quantity"] = "ten"
    dirty_df.at[22, "quantity"] = "N/A"
    dirty_df.at[55, "quantity"] = "abc"

    dirty_df["unit_price"] = dirty_df["unit_price"].astype(object)
    dirty_df.at[10, "unit_price"] = "$99.99"
    dirty_df.at[40, "unit_price"] = "free"

    # Formatting inconsistencies
    for idx in [0, 18, 37, 80, 130, 170]:
        val = dirty_df.at[idx, "sales_rep"]
        if pd.notna(val):
            dirty_df.at[idx, "sales_rep"] = f"  {val}  "

    return dirty_df, clean_df


# ---------------------------------------------------------------------------
# Hard dataset — HR, 500 rows  (stub — implemented in task 2.3)
# ---------------------------------------------------------------------------

def generate_hard() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dirty_df, clean_df) for the Hard HR dataset."""
    rng = np.random.default_rng(SEED + 2)

    n_clean = 480  # 480 unique rows; 20 duplicates added later

    departments = [
        "Engineering", "Marketing", "Sales", "HR", "Finance",
        "Operations", "Legal", "Support", "Design", "Product",
    ]
    titles = [
        "Engineer", "Manager", "Analyst", "Director", "Coordinator",
        "Specialist", "Associate", "Lead", "Intern", "Consultant",
    ]
    first_names = [
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank",
        "Iris", "Jack", "Karen", "Leo", "Mia", "Nate", "Olivia", "Paul",
        "Quinn", "Rachel", "Sam", "Tina",
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
        "White", "Harris", "Martin", "Thompson", "Young", "Allen",
    ]

    rows = []
    for i in range(n_clean):
        emp_id = 3001 + i
        fn = first_names[i % len(first_names)]
        ln = last_names[i % len(last_names)]
        name = f"{fn} {ln}"
        dept = departments[i % len(departments)]
        title = titles[i % len(titles)]
        salary = int(rng.integers(30000, 150000))
        years_exp = int(rng.integers(0, 30))
        year = int(rng.integers(2010, 2024))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        hire_date = f"{year}-{month:02d}-{day:02d}"
        email = f"{fn.lower()}.{ln.lower()}{emp_id}@company.com"
        rows.append({
            "employee_id": emp_id,
            "name": name,
            "department": dept,
            "title": title,
            "salary": salary,
            "years_experience": years_exp,
            "hire_date": hire_date,
            "email": email,
        })

    clean_df = pd.DataFrame(rows)
    dirty_df = clean_df.copy()

    # Duplicates
    dup_indices = list(range(0, 200, 10))[:20]
    duplicates = clean_df.iloc[dup_indices].copy()
    dirty_df = pd.concat([dirty_df, duplicates], ignore_index=True)

    # Missing values
    for col, idxs in [
        ("name", [5, 25, 80, 200, 350]),
        ("department", [10, 50, 120, 250, 400]),
        ("salary", [15, 60, 150, 300, 450]),
        ("email", [20, 70, 180, 320, 470]),
    ]:
        for idx in idxs:
            dirty_df.at[idx, col] = np.nan

    # Type errors
    dirty_df["salary"] = dirty_df["salary"].astype(object)
    for idx, val in [(8, "sixty thousand"), (35, "N/A"), (100, "abc"), (200, "$75000"), (300, "unknown")]:
        dirty_df.at[idx, "salary"] = val

    dirty_df["years_experience"] = dirty_df["years_experience"].astype(object)
    for idx, val in [(12, "five"), (45, "N/A"), (130, "abc")]:
        dirty_df.at[idx, "years_experience"] = val

    # Formatting inconsistencies
    for idx in [0, 30, 75, 150, 225, 300, 375, 450]:
        val = dirty_df.at[idx, "name"]
        if pd.notna(val):
            dirty_df.at[idx, "name"] = f"  {val}  "

    return dirty_df, clean_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating Easy dataset (Customer, 50 rows)...")
    easy_dirty, easy_clean = generate_easy()
    easy_dirty.to_csv(os.path.join(DATA_DIR, "easy_dirty.csv"), index=False)
    easy_clean.to_csv(os.path.join(DATA_DIR, "easy_clean.csv"), index=False)
    print(f"  easy_dirty.csv: {len(easy_dirty)} rows")
    print(f"  easy_clean.csv: {len(easy_clean)} rows")

    print("Generating Medium dataset (Sales, 200 rows)...")
    medium_dirty, medium_clean = generate_medium()
    medium_dirty.to_csv(os.path.join(DATA_DIR, "medium_dirty.csv"), index=False)
    medium_clean.to_csv(os.path.join(DATA_DIR, "medium_clean.csv"), index=False)
    print(f"  medium_dirty.csv: {len(medium_dirty)} rows")
    print(f"  medium_clean.csv: {len(medium_clean)} rows")

    print("Generating Hard dataset (HR, 500 rows)...")
    hard_dirty, hard_clean = generate_hard()
    hard_dirty.to_csv(os.path.join(DATA_DIR, "hard_dirty.csv"), index=False)
    hard_clean.to_csv(os.path.join(DATA_DIR, "hard_clean.csv"), index=False)
    print(f"  hard_dirty.csv: {len(hard_dirty)} rows")
    print(f"  hard_clean.csv: {len(hard_clean)} rows")

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
