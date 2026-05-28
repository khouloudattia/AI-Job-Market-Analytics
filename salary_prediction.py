# =========================================================
# 0. IMPORTS
# =========================================================
import pyodbc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest


# =========================================================
# 1. CONFIGURATION CONNEXION
# =========================================================
SERVER = "KHOULOUDSPC"
DATABASE = "JobsStaging"
QUERY = "SELECT * FROM dbo.fact_job_posts"


def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        "Trusted_Connection=yes;"
    )


# =========================================================
# 2. CHARGEMENT DATA
# =========================================================
conn = get_connection()
df = pd.read_sql(QUERY, conn)

print("Shape initiale:", df.shape)


# =========================================================
# 3. NETTOYAGE + FEATURE SALAIRE
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
# 4. TOP METIERS SALAIRE
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
job_salary = job_salary.sort_values("avg_salary", ascending=False)

print("\nTOP 20 métiers:")
print(job_salary.head(20))


# =========================================================
# 5. NLP EMBEDDINGS
# =========================================================
df_text = df[["job_title_short", "salary"]].copy()

nlp_model = SentenceTransformer("all-MiniLM-L6-v2")
X = nlp_model.encode(
    df_text["job_title_short"].tolist(),
    show_progress_bar=True
)

# =========================================================
# 6. PCA + NORMALISATION
# =========================================================
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)


# =========================================================
# 7. CHOIX AUTOMATIQUE DE K 
# =========================================================
k_values = range(2, 10)

inertias = []
silhouettes = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))


# =========================================================
# 8. VISUALISATION ELBOW + SILHOUETTE
# =========================================================

plt.figure()
plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.show()

plt.figure()
plt.plot(k_values, silhouettes, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show()


# =========================================================
# 9. CHOIX DU MEILLEUR K
# =========================================================
best_k = k_values[np.argmax(silhouettes)]
print("\nBest K choisi:", best_k)


# =========================================================
# 10. CLUSTERING FINAL
# =========================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_text["cluster"] = kmeans.fit_predict(X_scaled)


# =========================================================
# 11. ANALYSE CLUSTERS
# =========================================================
cluster_salary = df_text.groupby("cluster").agg(
    avg_salary=("salary", "mean"),
    median_salary=("salary", "median"),
    count=("salary", "count")
).reset_index()

print("\nSALAIRES PAR CLUSTER")
print(cluster_salary.sort_values("avg_salary", ascending=False))


# Silhouette score final (très important entretien)
final_score = silhouette_score(X_scaled, df_text["cluster"])
print("\nSilhouette Score final:", final_score)


# =========================================================
# 12. VISUALISATIONS
# =========================================================
# Cluster salary
plt.figure(figsize=(8,5))
plt.bar(cluster_salary["cluster"].astype(str),
        cluster_salary["avg_salary"])
plt.title("Average Salary by Cluster")
plt.show()



# =========================================================
# ANOMALY DETECTION 
# =========================================================

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

df_anom = df.copy()

# =========================================================
# 1. FEATURE ENGINEERING SALAIRE
# =========================================================
df_anom["salary"] = df_anom["salary_year_avg"]
df_anom["salary"] = df_anom["salary"].fillna(df_anom["salary_hour_avg"] * 160 * 12)

df_anom = df_anom.dropna(subset=["salary", "job_title_short"])
df_anom = df_anom[df_anom["salary"] > 0]

df_anom["job_title_short"] = df_anom["job_title_short"].str.lower().str.strip()

df_anom["anomaly"] = 1


# =========================================================
# 2. GLOBAL SCALER 
# =========================================================
scaler = StandardScaler()


# =========================================================
# 3. DETECTION PAR METIER
# =========================================================
for job in df_anom["job_title_short"].unique():

    df_job = df_anom[df_anom["job_title_short"] == job]

    if len(df_job) < 50:
        continue

    X = scaler.fit_transform(df_job[["salary"]])

    iso_model = IsolationForest(
        contamination="auto", 
        random_state=42
    )

    preds = iso_model.fit_predict(X)

    df_anom.loc[df_job.index, "anomaly"] = preds


# =========================================================
# 4. STATISTIQUES ANOMALIES
# =========================================================
total_anomalies = (df_anom["anomaly"] == -1).sum()

print("\nTotal anomalies:", total_anomalies)

print("\nAnomalies par métier:")
print(df_anom[df_anom["anomaly"] == -1]["job_title_short"].value_counts().head(10))


# =========================================================
# 5. VISUALISATION : ANOMALIES PAR METIER
# =========================================================
import matplotlib.pyplot as plt

anomaly_counts = df_anom[df_anom["anomaly"] == -1]["job_title_short"].value_counts()

plt.figure(figsize=(12,6))
anomaly_counts.plot(kind="bar")

plt.title("Number of Anomalies per Job Title")
plt.xlabel("Job Title")
plt.ylabel("Count of anomalies")
plt.xticks(rotation=90)

plt.show()


# =========================================================
# SMART JOB SALARY LOOKUP (NLP SEARCH)
# =========================================================

from sklearn.metrics.pairwise import cosine_similarity

print("SMART JOB SALARY LOOKUP")

# =========================================================
# 1. JOB TITLES UNIQUES
# =========================================================
unique_jobs = job_salary["job_title_short"].unique().tolist()

# =========================================================
# 2. EMBEDDINGS JOB TITLES
# =========================================================
job_embeddings = nlp_model.encode(unique_jobs)

# =========================================================
# 3. INPUT UTILISATEUR
# =========================================================
user_job = str(
    input("\nEnter a job title: ")
).lower().strip()

# =========================================================
# 4. EMBEDDING INPUT UTILISATEUR
# =========================================================
user_embedding = nlp_model.encode([user_job])

# =========================================================
# 5. CALCUL SIMILARITÉ
# =========================================================
similarities = cosine_similarity(user_embedding, job_embeddings)

best_match_index = np.argmax(similarities)
best_match_job = unique_jobs[best_match_index]

similarity_score = similarities[0][best_match_index]

# =========================================================
# 6. TOP SIMILAR JOBS 
# =========================================================
top_indices = similarities[0].argsort()[-5:][::-1]

# =========================================================
# 7. RÉCUPÉRATION INFOS SALAIRE
# =========================================================
job_info = job_salary[
    job_salary["job_title_short"] == best_match_job
]

avg_salary = job_info["avg_salary"].values[0]
median_salary = job_info["median_salary"].values[0]
job_count = job_info["count"].values[0]

# =========================================================
# 8. SALARY LEVEL 
# =========================================================
if avg_salary > 130000:
    salary_level = "High Paying"

elif avg_salary > 100000:
    salary_level = "Medium Paying"

else:
    salary_level = "Standard Paying"

# =========================================================
# 9. CLUSTER DU MÉTIER
# =========================================================
cluster_info = df_text[
    df_text["job_title_short"] == best_match_job
]["cluster"].mode()[0]

# =========================================================
# 10. NOMBRE D'ANOMALIES
# =========================================================
anomaly_count = len(
    df_anom[
        (df_anom["job_title_short"] == best_match_job) &
        (df_anom["anomaly"] == -1)
    ]
)

# =========================================================
# 11. ANOMALY RATE 
# =========================================================
anomaly_rate = anomaly_count / job_count

# =========================================================
# 12. AFFICHAGE RÉSULTATS
# =========================================================
print("\n=========== RESULT ===========")

print(f"\nUser input            : {user_job}")
print(f"Closest match found   : {best_match_job}")
print(f"Similarity score      : {similarity_score:.2f}")

print(f"\nAverage Salary        : ${avg_salary:,.0f}")
print(f"Median Salary         : ${median_salary:,.0f}")
print(f"Number of Jobs        : {job_count}")

print(f"\nSalary Level          : {salary_level}")

print(f"\nCluster ID            : {cluster_info}")

print(f"\nDetected Anomalies    : {anomaly_count}")
print(f"Anomaly Rate          : {anomaly_rate:.2%}")

# =========================================================
# 13. TOP SIMILAR JOBS AFFICHAGE 
# =========================================================
print("\nTop Similar Jobs:")

for idx in top_indices:
    print(f"- {unique_jobs[idx]}")



# =========================================================
# 14. EXPORT CSV FOR POWER BI 
# =========================================================

import os

# dossier du script 
output_dir = r"C:\Users\khoul\OneDrive\Desktop\Mission d'entreprise"

# sécurité : créer dossier si besoin
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# EXPORT FILES
# =========================================================
job_salary.to_csv(f"{output_dir}/job_salary_analysis.csv", index=False)

df_anom.to_csv(f"{output_dir}/anomaly_results.csv", index=False)

df_text.to_csv(f"{output_dir}/cluster_results.csv", index=False)

# =========================================================
# DEBUG 
# =========================================================
print("\nCSV files exported successfully!")
print("Saved in:", output_dir)

print("\nFiles created:")
print("- job_salary_analysis.csv")
print("- anomaly_results.csv")
print("- cluster_results.csv")


# =========================================================
# WORK MODE ANALYSIS 
# =========================================================

print("WORK MODE ANALYSIS")

selected_job = best_match_job
print(f"\nSelected Job: {selected_job}")

df_job_mode = df[df["job_title_short"] == selected_job].copy()

if df_job_mode.empty or "job_schedule_type" not in df_job_mode.columns:
    print("\nNo work mode data available for this job.")

else:

    # =========================================================
    # 1. AUTO-DETECT SALARY COLUMN 
    # =========================================================
    if "avg_salary" in df_job_mode.columns:
        salary_col = "avg_salary"
    elif "salary" in df_job_mode.columns:
        salary_col = "salary"
    elif "median_salary" in df_job_mode.columns:
        salary_col = "median_salary"
    else:
        raise ValueError("No salary column found in dataset")

    # =========================================================
    # 2. CLEAN DATA
    # =========================================================
    df_job_mode = df_job_mode.dropna(subset=["job_schedule_type", salary_col])

    df_job_mode["job_schedule_type"] = (
        df_job_mode["job_schedule_type"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # =========================================================
    # 3. MODE STATS
    # =========================================================
    mode_stats = (
        df_job_mode.groupby("job_schedule_type")
        .agg(
            job_count=("job_id", "count"),
            avg_salary=(salary_col, "mean"),
            median_salary=(salary_col, "median")
        )
        .reset_index()
        .sort_values("job_count", ascending=False)
    )

    # =========================================================
    # 4. BEST MODE
    # =========================================================
    best_mode_salary = mode_stats.sort_values("avg_salary", ascending=False).head(1)

    # =========================================================
    # 5. GAP
    # =========================================================
    salary_gap = mode_stats["avg_salary"].max() - mode_stats["avg_salary"].min()

    # =========================================================
    # 6. OUTPUT
    # =========================================================
    print("\nWork Mode Distribution:")
    print(mode_stats)

    print("\nBest Paying Work Mode:")
    print(best_mode_salary)

    print(f"\nSalary Gap between modes: ${salary_gap:,.0f}")

    # =========================================================
    # 7. SAFE INSIGHT
    # =========================================================
    if not best_mode_salary.empty:
        best_mode = best_mode_salary["job_schedule_type"].values[0]
    else:
        best_mode = "N/A"

    print("\nInsight:")
    print(f"- Best work mode for salary: {best_mode}")
    print("- Salary variability depends on work flexibility level")









# =========================================================
# SMART SKILL RECOMMENDATION 
# =========================================================

print("SMART SKILL RECOMMENDATION")

# =========================================================
# 1. SELECTED JOB (NLP RESULT)
# =========================================================
selected_job = best_match_job
print(f"\nSelected Job: {selected_job}")

# =========================================================
# 2. GET JOB KEYS
# =========================================================
job_keys = df[df["job_title_short"] == selected_job]["job_key"].unique()

if len(job_keys) == 0:
    print("\nNo job_key found for selected job.")

else:

        # =========================================================
        # 3. LOAD BRIDGE TABLE
        # =========================================================
        bridge_job_skill = pd.read_sql("""
            SELECT job_key, skill_key
            FROM dbo.bridge_job_skill
        """, conn)

        # =========================================================
        # 4. LOAD DIM SKILL TABLE
        # =========================================================
        dim_skill = pd.read_sql("""
            SELECT skill_key, skill_name, skill_type
            FROM dbo.dim_skill
        """, conn)

        # =========================================================
        # 5. FILTER BRIDGE
        # =========================================================
        job_skill_map = bridge_job_skill[
            bridge_job_skill["job_key"].isin(job_keys)
        ]

        # =========================================================
        # 6. JOIN WITH DIM SKILL
        # =========================================================
        job_skill_full = job_skill_map.merge(
            dim_skill,
            on="skill_key",
            how="left"
        )

        # =========================================================
        # 7. SAFETY CHECK
        # =========================================================
        if job_skill_full.empty:
            print("\nNo skills found for this job.")

        else:

            # =========================================================
            # 8. CLEAN DATA
            # =========================================================
            job_skill_full["skill_name"] = (
                job_skill_full["skill_name"]
                .fillna("Unknown")
                .str.strip()
            )

            job_skill_full["skill_type"] = (
                job_skill_full["skill_type"]
                .fillna("HARD")
                .str.upper()
                .str.strip()
            )

            # =========================================================
            # 9. SKILL RANKING
            # =========================================================
            skill_rank = (
                job_skill_full["skill_name"]
                .value_counts()
                .reset_index()
            )
            skill_rank.columns = ["skill", "frequency"]

            # =========================================================
            # 10. HARD SKILLS
            # =========================================================
            hard_skills = (
                job_skill_full[job_skill_full["skill_type"] == "HARD"]
                ["skill_name"]
                .value_counts()
                .head(10)
            )

            # =========================================================
            # 11. SOFT SKILLS 
            # =========================================================
            soft_skills_keywords = [
                "communication",
                "teamwork",
                "leadership",
                "problem solving",
                "collaboration"
            ]

            job_skill_full["is_soft"] = job_skill_full["skill_name"].apply(
                lambda x: any(k.lower() in str(x).lower() for k in soft_skills_keywords)
            )

            soft_skills = (
                job_skill_full[job_skill_full["is_soft"] == True]
                ["skill_name"]
                .value_counts()
                .head(10)
            )

            # fallback si vide
            if soft_skills.empty:
                soft_skills = ["Communication", "Teamwork", "Problem Solving"]
# =========================================================
# 12. OUTPUT RESULTS
# =========================================================

print("\nHARD SKILLS (Most Required)")

for skill, count in hard_skills.items():
    print(f"- {skill}: {count}")

print("\nSOFT SKILLS")

# si fallback list
if isinstance(soft_skills, list):

    for skill in soft_skills:
        print(f"- {skill}")

# si pandas Series
else:

    for skill, count in soft_skills.items():
        print(f"- {skill}: {count}")

# =========================================================
# 13. CAREER INSIGHT
# =========================================================
print("\nKey Skills to Learn:")

for s in skill_rank.head(5)["skill"]:
    print(f"- {s}")
 
df_lookup = pd.DataFrame([{
    "user_job": user_job,
    "best_match_job": best_match_job,
    "similarity_score": similarity_score
}])

df_lookup.to_csv(f"{output_dir}/job_lookup.csv", index=False)