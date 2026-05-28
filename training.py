# =========================================================
# IMPORTS
# =========================================================
import os
import pyodbc
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONNECTION
# =========================================================
SERVER = "KHOULOUDSPC"
DATABASE = "JobsStaging"


def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
    )


# =========================================================
# LOAD DATA
# =========================================================
conn = get_connection()
df = pd.read_sql("SELECT * FROM dbo.fact_job_posts", conn)


# =========================================================
# CLEAN DATA
# =========================================================
def build_salary(df):
    df = df.copy()
    df["salary"] = df["salary_year_avg"]
    df["salary"] = df["salary"].fillna(df["salary_hour_avg"] * 160 * 12)
    df = df.dropna(subset=["salary", "job_title_short"])
    df = df[df["salary"] > 0]
    df["job_title_short"] = df["job_title_short"].str.lower().str.strip()
    return df


df = build_salary(df)


# =========================================================
# JOB STATISTICS
# =========================================================
job_salary = (
    df.groupby("job_title_short")
    .agg(
        avg_salary=("salary", "mean"),
        median_salary=("salary", "median"),
        count=("salary", "count")
    )
    .reset_index()
)

job_salary = job_salary[job_salary["count"] >= 50]


# =========================================================
# NLP MODEL
# =========================================================
nlp_model = SentenceTransformer("all-MiniLM-L6-v2")

unique_jobs = job_salary["job_title_short"].unique().tolist()
job_embeddings = nlp_model.encode(unique_jobs)


# =========================================================
# CREATE MODELS FOLDER
# =========================================================
os.makedirs("models", exist_ok=True)


# =========================================================
# SAVE MODELS (CLEAN VERSION)
# =========================================================
def save_models():
    joblib.dump(job_salary, "models/job_salary.pkl")
    joblib.dump(unique_jobs, "models/unique_jobs.pkl")
    joblib.dump(job_embeddings, "models/job_embeddings.pkl")
    joblib.dump(nlp_model, "models/nlp_model.pkl")

    print("✅ Models saved successfully in /models")


# =========================================================
# RUN EXPORT ONLY IF FILE EXECUTED DIRECTLY
# =========================================================
if __name__ == "__main__":
    save_models()