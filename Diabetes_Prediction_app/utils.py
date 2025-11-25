import sqlite3
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import os

DB_DIR = Path("db")
DB_PATH = DB_DIR / "predictions.db"

def init_db(db_path: Path = DB_PATH):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            gender TEXT,
            age REAL,
            height_cm REAL,
            weight_kg REAL,
            bmi REAL,
            bp TEXT,
            glucose REAL,
            prediction INTEGER,
            probability REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(record: Dict[str, Any], db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Combine systolic/diastolic into single BP field
    bp_value = f"{record.get('systolic', 0)}/{record.get('diastolic', 0)}"

    c.execute('''
        INSERT INTO predictions(
            gender, age, height_cm, weight_kg, bmi, bp, glucose, prediction, probability
        ) VALUES (?,?,?,?,?,?,?,?,?)
    ''', (
        record.get('gender'),
        float(record.get('age', 0)),
        float(record.get('height_cm', 0)),
        float(record.get('weight_kg', 0)),
        float(record.get('bmi', 0)),
        bp_value,
        float(record.get('glucose', 0)),
        int(record.get('prediction', 0)),
        float(record.get('probability', 0.0))
    ))

    conn.commit()
    conn.close()

def load_history(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
    conn.close()
    return df
