from flask import Flask, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
import os

app = Flask(__name__)

# =========================================================
# LOAD MODELS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

job_salary = joblib.load(os.path.join(MODEL_DIR, "job_salary.pkl"))
unique_jobs = joblib.load(os.path.join(MODEL_DIR, "unique_jobs.pkl"))
job_embeddings = joblib.load(os.path.join(MODEL_DIR, "job_embeddings.pkl"))
nlp_model = joblib.load(os.path.join(MODEL_DIR, "nlp_model.pkl"))

# =========================================================
# ANOMALY DETECTION MODEL
# =========================================================
salary_values = job_salary["avg_salary"].values.reshape(-1, 1)

anomaly_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

job_salary["anomaly"] = anomaly_model.fit_predict(salary_values)

job_salary["anomaly_score"] = anomaly_model.decision_function(
    salary_values
)

# =========================================================
# GLOBAL STATE (download)
# =========================================================
last_result = {}

# =========================================================
# GLOBAL DASHBOARD DATA
# =========================================================
salary_dashboard = {
    "min": round(float(job_salary["avg_salary"].min()), 2),
    "max": round(float(job_salary["avg_salary"].max()), 2),
    "global_avg": round(float(job_salary["avg_salary"].mean()), 2)
}

# =========================================================
# TOP JOBS FOR CHART
# =========================================================
top_jobs = (
    job_salary
    .sort_values("avg_salary", ascending=False)
    .head(8)
)

job_names = top_jobs["job_title_short"].tolist()

salary_distribution = (
    top_jobs["avg_salary"]
    .round(2)
    .tolist()
)

# =========================================================
# SKILLS DATABASE (STATIC DEMO)
# =========================================================
skills_map = {
    "data scientist": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "SQL",
        "Pandas"
    ],

    "data analyst": [
        "Excel",
        "SQL",
        "Power BI",
        "Python",
        "Statistics"
    ],

    "machine learning engineer": [
        "TensorFlow",
        "PyTorch",
        "Python",
        "NLP",
        "MLOps"
    ],

    "software engineer": [
        "Java",
        "Python",
        "Git",
        "APIs",
        "Algorithms"
    ]
}

# =========================================================
# HOME (RESET RESULTS ON REFRESH)
# =========================================================
@app.route("/")
def home():

    global last_result

    # reset page state
    last_result = {}

    return render_template(
        "index.html",
        result=None
    )

# =========================================================
# PREDICT ROUTE
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():

    global last_result

    # =====================================================
    # USER INPUT
    # =====================================================
    user_job = request.form["job"].lower().strip()

    # =====================================================
    # NLP EMBEDDING
    # =====================================================
    user_emb = nlp_model.encode([user_job])

    similarities = cosine_similarity(
        user_emb,
        job_embeddings
    )

    best_idx = np.argmax(similarities)

    best_job = unique_jobs[best_idx]

    # =====================================================
    # JOB INFO
    # =====================================================
    job_info = job_salary[
        job_salary["job_title_short"] == best_job
    ]

    avg_salary = round(
        float(job_info["avg_salary"].values[0]), 2
    )

    median_salary = round(
        float(job_info["median_salary"].values[0]), 2
    )

    count = int(job_info["count"].values[0])

    # =====================================================
    # ANOMALY
    # =====================================================
    anomaly_flag = int(job_info["anomaly"].values[0])

    anomaly_score = round(
        float(job_info["anomaly_score"].values[0]),
        4
    )

    # =====================================================
    # WORK MODE ANALYSIS
    # =====================================================
    if avg_salary > 150000:
        work_mode = "Remote / International"
    elif avg_salary > 90000:
        work_mode = "Hybrid"
    else:
        work_mode = "Onsite"

    # =====================================================
    # MARKET LEVEL
    # =====================================================
    if avg_salary > 180000:
        market_level = "Elite Market"
    elif avg_salary > 120000:
        market_level = "High Demand"
    else:
        market_level = "Standard Market"

    # =====================================================
    # SKILLS
    # =====================================================
    hard_skills = skills_map.get(
        best_job,
        ["Python", "SQL", "Analytics"]
    )

    soft_skills = [
        "Communication",
        "Problem Solving",
        "Teamwork",
        "Critical Thinking"
    ]

    # =====================================================
    # RESULT
    # =====================================================
    result = {
        "input": user_job,
        "best_match": best_job,

        "avg_salary": avg_salary,
        "median_salary": median_salary,
        "count": count,

        "similarity": round(
            float(similarities[0][best_idx]),
            4
        ),

        "anomaly": anomaly_flag,
        "anomaly_score": anomaly_score,

        "work_mode": work_mode,
        "market_level": market_level
    }

    # =====================================================
    # SAVE FOR DOWNLOAD
    # =====================================================
    last_result = result

    # =====================================================
    # RETURN TEMPLATE
    # =====================================================
    return render_template(
        "index.html",

        result=result,

        salary_dashboard=salary_dashboard,

        job_names=job_names,

        salary_distribution=salary_distribution,

        hard_skills=hard_skills,

        soft_skills=soft_skills
    )

# =========================================================
# DOWNLOAD REPORT
# =========================================================
@app.route("/download")
def download():

    global last_result

    if not last_result:
        return "No data available"

    df = pd.DataFrame([last_result])

    file_path = os.path.join(
        BASE_DIR,
        "report.csv"
    )

    df.to_csv(file_path, index=False)

    return send_file(
        file_path,
        as_attachment=True
    )

# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)



