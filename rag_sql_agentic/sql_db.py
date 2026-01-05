import pandas as pd
import sqlite3
import os
from pathlib import Path

def convert_csv_to_sqlite(csv_path: str, db_path: str, table_name: str = "hospital_data"):
    """
    Reads a CSV file and writes it to a SQLite database.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    print(f"Reading CSV from {csv_path}...")
    # Read CSV
    # Low memory=False because mixed types might occur, or let pandas infer
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Clean column names: replace spaces with underscores, lowercase
    df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower() for c in df.columns]
    
    print(f"Columns: {df.columns.tolist()}")
    
    print(f"Writing to SQLite db at {db_path} in table '{table_name}'...")
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    
    print("Conversion complete.")
