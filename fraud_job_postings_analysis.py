# %% [markdown]
# # Detecting Fraudulent Job Postings Using Text Analytics
# **INSY 669/670 — Group Project**
# Team: Julia Al Rifai, Katerina Avvakumova, Mia Ghrayeb, Daniel Puente Cavazos, Alex Zhao
#
# **Business question.** Job platforms receive too many postings for moderators to inspect each one.
# Can text analytics produce a *fraud-risk score* that ranks postings so moderators review the riskiest first?
#
# **Research questions**
# 1. Where do the fraud signals come from: the posting **text**, the structured **metadata**, or both combined?
# 2. Do more complex models meaningfully beat an interpretable TF-IDF + logistic regression baseline once tuned?
# 3. What language distinguishes fraudulent postings from legitimate ones?
# 4. Without using labels, do distinct *types* of fraudulent postings emerge (topic modeling)?
# 5. In practice, how much fraud does a moderator catch by reviewing only the top X% riskiest postings?
#
# **Notebook structure**
# 1. Setup · 2. Data import & description · 3. Preprocessing & feature engineering · 4. Exploratory analysis ·
# 5. Experiment 1 — text vs metadata vs combined · 6. Experiment 2 — model comparison & hyperparameter tuning ·
# 7. Interpretation — what fraud language looks like · 8. Unsupervised — topic modeling (NMF) ·
# 9. Fraud-risk score & review-budget analysis · 10. Error analysis · 11. Summary of results
#
# **Data:** Employment Scam Aegean Dataset (EMSCAD), Kaggle "Real or Fake Job Posting Prediction"
# (`fake_job_postings.csv`, ~17,880 postings). Place the CSV in `data/` (see README).

# %% [markdown]
# ## 1. Setup

# %%
import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import NMF
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
DATA_PATH = Path("data/fake_job_postings.csv")
FIG_DIR = Path("figures");  FIG_DIR.mkdir(exist_ok=True)   # PNGs for the report and slides
RES_DIR = Path("results");  RES_DIR.mkdir(exist_ok=True)   # CSV / JSON result tables
REVIEW_BUDGET = 0.05   # headline operating point: moderators review the top 5% riskiest postings
N_JOBS = -1

FRAUD_C, REAL_C, GREY_C = "#c0392b", "#2c7fb8", "#9e9e9e"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})

def savefig(name):
    """Save the current figure to figures/ (used in the report and slide deck)."""
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 2. Data import & description

# %%
df = pd.read_csv(DATA_PATH)
print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
df.head(3)

# %%
# Data description table: type, % missing and an example value for every variable.
desc = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "pct_missing": (df.isna().mean() * 100).round(1),
    "n_unique": df.nunique(),
    "example": df.apply(lambda c: str(c.dropna().iloc[0])[:60] if c.notna().any() else ""),
})
desc.to_csv(RES_DIR / "data_description.csv")
desc

# %%
n_fraud = int(df["fraudulent"].sum())
print(f"Fraudulent postings: {n_fraud:,} of {len(df):,} ({n_fraud/len(df):.1%})")
print("=> Heavy class imbalance: a model that predicts 'legitimate' for everything is "
      f"{1 - n_fraud/len(df):.1%} accurate, so accuracy is NOT a useful metric here.")

# %% [markdown]
# ## 3. Preprocessing & feature engineering
#
# **Text cleaning.** EMSCAD replaces URLs, emails and phone numbers with hashed placeholders
# (e.g. `#URL_86fd…#`). We map them to tokens (`urltoken`, `emailtoken`, `phonetoken`) so they can act as
# signals instead of thousands of unique hashes. HTML stripping in the original data also glued words together
# (`experienceWe`), so we split lower→upper case boundaries before lower-casing.
#
# **Feature groups** (kept separate so Experiment 1 can isolate each signal):
# * **TEXT** — TF-IDF over title + company profile + description + requirements + benefits, plus simple text
#   statistics (word counts, placeholder counts).
# * **META** — structured fields only: binary flags (`telecommuting`, `has_company_logo`, `has_questions`),
#   salary provided, which fields were left empty, and categorical fields (employment type, experience,
#   education, industry, function, country).

# %%
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
CAT_COLS = ["employment_type", "required_experience", "required_education", "industry", "function", "country"]
BIN_COLS = ["telecommuting", "has_company_logo", "has_questions"]
MISSING_FIELDS = ["company_profile", "requirements", "benefits", "department", "salary_range",
                  "industry", "function", "required_education", "required_experience", "employment_type"]

PLACEHOLDERS = {"URL": " urltoken ", "EMAIL": " emailtoken ", "PHONE": " phonetoken "}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    for key, tok in PLACEHOLDERS.items():
        s = re.sub(rf"#{key}_[^#\s]*#", tok, s)
    s = re.sub(r"<[^>]+>", " ", s)                 # leftover HTML tags
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)     # split glued words: "experienceWe" -> "experience We"
    s = s.lower()
    s = re.sub(r"[^a-z0-9$€£%\s]", " ", s)         # keep currency / % symbols, drop other punctuation
    return re.sub(r"\s+", " ", s).strip()

data = df.copy()
for col in TEXT_FIELDS:
    data[f"{col}_clean"] = data[col].apply(clean_text)
data["text_all"] = data[[f"{c}_clean" for c in TEXT_FIELDS]].agg(" ".join, axis=1)

# --- META features ---
data["country"] = data["location"].fillna("").str.split(",").str[0].str.strip().replace("", np.nan)
data["has_salary"] = data["salary_range"].notna().astype(int)
for col in MISSING_FIELDS:
    data[f"missing_{col}"] = data[col].isna().astype(int)
for col in CAT_COLS:
    data[col] = data[col].fillna("Missing").astype(str)
META_NUM_COLS = BIN_COLS + ["has_salary"] + [f"missing_{c}" for c in MISSING_FIELDS]

# --- TEXT statistics (derived from text, so they belong to the TEXT group) ---
for col in ["company_profile", "description", "requirements", "benefits"]:
    data[f"len_{col}"] = np.log1p(data[f"{col}_clean"].str.split().str.len())
raw_all = df[TEXT_FIELDS].fillna("").agg(" ".join, axis=1)
for key in PLACEHOLDERS:
    data[f"n_{key.lower()}"] = raw_all.str.count(rf"#{key}_")
data["n_caps_words"] = raw_all.str.count(r"\b[A-Z]{3,}\b")       # shouting ("URGENT", "EARN")
data["n_money_symbols"] = raw_all.str.count(r"[$€£]")
TEXT_STAT_COLS = [f"len_{c}" for c in ["company_profile", "description", "requirements", "benefits"]] + \
                 [f"n_{k.lower()}" for k in PLACEHOLDERS] + ["n_caps_words", "n_money_symbols"]

y = data["fraudulent"].astype(int).values
print("Example cleaned text:\n", data.loc[0, "text_all"][:400], "...")

# %% [markdown]
# ## 4. Exploratory analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

counts = data["fraudulent"].value_counts().sort_index()
axes[0].bar(["Legitimate", "Fraudulent"], counts.values, color=[REAL_C, FRAUD_C])
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}\n({v/len(data):.1%})", ha="center", va="bottom")
axes[0].set_title("Class balance"); axes[0].set_ylabel("Postings"); axes[0].set_ylim(0, counts.max() * 1.2)

flags = {"No company logo": data["has_company_logo"] == 0,
         "No company profile": data["missing_company_profile"] == 1,
         "Salary listed": data["has_salary"] == 1,
         "Telecommuting": data["telecommuting"] == 1,
         "No screening questions": data["has_questions"] == 0}
base = data["fraudulent"].mean()
rates = pd.Series({k: data.loc[m, "fraudulent"].mean() for k, m in flags.items()}).sort_values()
axes[1].barh(rates.index, rates.values * 100, color=FRAUD_C)
axes[1].axvline(base * 100, color="black", ls="--", lw=1)
axes[1].text(base * 100, len(rates) - 0.4, f" overall {base:.1%}", va="center")
axes[1].set_xlabel("% fraudulent among postings with this property")
axes[1].set_title("Fraud rate by posting metadata")
savefig("01_eda_class_balance_and_metadata")

# %%
# Fraud rate by industry (industries with at least 50 postings)
ind = data.groupby("industry")["fraudulent"].agg(["mean", "size"]).query("size >= 50")
ind = ind.sort_values("mean", ascending=False).head(12)
plt.figure(figsize=(8, 4))
plt.barh(ind.index[::-1], ind["mean"][::-1] * 100, color=FRAUD_C)
plt.axvline(base * 100, color="black", ls="--", lw=1)
plt.xlabel("% fraudulent"); plt.title("Industries with the highest fraud rate (n ≥ 50)")
savefig("02_eda_fraud_by_industry")

# %%
# Text length: fraudulent postings tend to have thinner company profiles
plt.figure(figsize=(8, 3.5))
for label, color, name in [(0, REAL_C, "Legitimate"), (1, FRAUD_C, "Fraudulent")]:
    words = data.loc[data["fraudulent"] == label, "company_profile_clean"].str.split().str.len()
    plt.hist(words.clip(upper=400), bins=40, alpha=0.55, density=True, color=color, label=name)
plt.xlabel("Words in company profile (capped at 400)"); plt.ylabel("Density")
plt.title("Company profile length"); plt.legend()
savefig("03_eda_company_profile_length")

# %% [markdown]
# ## 5. Train/test split and evaluation protocol
#
# * Stratified **80/20 split**; the test set is held out and used **only once** per final model.
# * All model selection and tuning uses **5-fold stratified cross-validation on the training set**.
# * **Primary metric: PR-AUC (average precision)** — suited to rare positives.
#   Also reported: ROC-AUC, and **Recall@5%** = share of all fraud caught if moderators review the 5% highest-risk
#   postings (the operational metric for a platform), and Precision@5%.

# %%
FEATURE_COLS = ["text_all"] + CAT_COLS + META_NUM_COLS + TEXT_STAT_COLS
X_train, X_test, y_train, y_test = train_test_split(
    data[FEATURE_COLS], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
train_text = data.loc[X_train.index]
print(f"Train: {len(X_train):,} ({y_train.mean():.1%} fraud)   Test: {len(X_test):,} ({y_test.mean():.1%} fraud)")

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def recall_at_budget(y_true, score, budget=REVIEW_BUDGET):
    k = max(1, int(round(budget * len(score))))
    top = np.argsort(-score)[:k]
    return y_true[top].sum() / y_true.sum()

def precision_at_budget(y_true, score, budget=REVIEW_BUDGET):
    k = max(1, int(round(budget * len(score))))
    return y_true[np.argsort(-score)[:k]].mean()

def evaluate(y_true, score):
    return {"PR-AUC": average_precision_score(y_true, score),
            "ROC-AUC": roc_auc_score(y_true, score),
            f"Recall@{REVIEW_BUDGET:.0%}": recall_at_budget(y_true, score),
            f"Precision@{REVIEW_BUDGET:.0%}": precision_at_budget(y_true, score)}

def get_scores(model, X):
    """Continuous risk score from any fitted classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def make_features(kind="combined", ngram_range=(1, 2), nonneg=False, max_features=50_000):
    """ColumnTransformer for one feature group: 'text', 'meta' or 'combined'.
    nonneg=True swaps StandardScaler for MinMaxScaler (needed by Naive Bayes)."""
    scaler = MinMaxScaler() if nonneg else StandardScaler()
    text = [("tfidf", TfidfVectorizer(ngram_range=ngram_range, min_df=3, max_df=0.9, sublinear_tf=True,
                                      stop_words="english", max_features=max_features), "text_all"),
            ("text_stats", scaler, TEXT_STAT_COLS)]
    meta = [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CAT_COLS),
            ("flags", "passthrough", META_NUM_COLS)]
    parts = {"text": text, "meta": meta, "combined": text + meta}[kind]
    return ColumnTransformer(parts, sparse_threshold=1.0)

# %% [markdown]
# ## 6. Experiment 1 — Where do the fraud signals come from?
# Same interpretable baseline (TF-IDF + logistic regression, class-balanced) trained on three feature sets.
# Cross-validated on the training set, then scored once on the test set.

# %%
exp1_rows, exp1_scores = [], {}
for kind in ["meta", "text", "combined"]:
    pipe = Pipeline([("features", make_features(kind)),
                     ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000,
                                                solver="liblinear"))])
    t0 = time.time()
    cv = cross_validate(pipe, X_train, y_train, cv=CV, scoring=["average_precision", "roc_auc"], n_jobs=N_JOBS)
    pipe.fit(X_train, y_train)
    s = get_scores(pipe, X_test); exp1_scores[kind] = s
    exp1_rows.append({"features": kind,
                      "CV PR-AUC": cv["test_average_precision"].mean(),
                      "CV PR-AUC sd": cv["test_average_precision"].std(),
                      "CV ROC-AUC": cv["test_roc_auc"].mean(),
                      **{f"Test {k}": v for k, v in evaluate(y_test, s).items()},
                      "seconds": round(time.time() - t0, 1)})
exp1 = pd.DataFrame(exp1_rows).set_index("features")
exp1.to_csv(RES_DIR / "exp1_text_vs_meta.csv")
exp1.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
labels = {"meta": "Metadata only", "text": "Text only", "combined": "Text + metadata"}
colors = {"meta": GREY_C, "text": REAL_C, "combined": FRAUD_C}
axes[0].bar([labels[k] for k in exp1.index], exp1["CV PR-AUC"], yerr=exp1["CV PR-AUC sd"],
            color=[colors[k] for k in exp1.index], capsize=4)
for i, v in enumerate(exp1["CV PR-AUC"]):
    axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center")
axes[0].axhline(y_train.mean(), color="black", ls="--", lw=1)
axes[0].text(2.4, y_train.mean(), "random", va="bottom", ha="right", fontsize=8)
axes[0].set_ylim(0, 1); axes[0].set_ylabel("PR-AUC (5-fold CV)"); axes[0].set_title("Signal source comparison")
for kind, s in exp1_scores.items():
    p, r, _ = precision_recall_curve(y_test, s)
    axes[1].plot(r, p, color=colors[kind], lw=2, label=f"{labels[kind]} (AP={average_precision_score(y_test, s):.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision–recall curves (test set)")
axes[1].legend(loc="lower left")
savefig("04_exp1_text_vs_metadata")

# %% [markdown]
# ## 7. Experiment 2 — Model comparison & hyperparameter tuning
# All models use the combined feature set. Each is tuned with `GridSearchCV` (5-fold, optimising PR-AUC).
# Tuned choices include text representation (unigrams vs uni+bigrams), regularisation strength, and
# **how class imbalance is handled** (`class_weight=None` vs `balanced`).
#
# | Model | Why include it | Grid |
# |---|---|---|
# | Logistic regression | Interpretable baseline from the proposal | n-grams × C × class weight |
# | Linear SVM | Strong standard for sparse text | C × class weight |
# | Complement Naive Bayes | Classic text model designed for imbalanced classes | alpha |
# | Random forest | Non-linear; can capture metadata interactions | depth/leaf size × class weight |

# %%
def tfidf_param(name):
    return f"features__tfidf__{name}"

searches = {
    "Logistic regression": (
        Pipeline([("features", make_features("combined")),
                  ("clf", LogisticRegression(max_iter=3000, solver="liblinear"))]),
        {tfidf_param("ngram_range"): [(1, 1), (1, 2)],
         "clf__C": [0.1, 1, 10],
         "clf__class_weight": [None, "balanced"]}),
    "Linear SVM": (
        Pipeline([("features", make_features("combined")), ("clf", LinearSVC(max_iter=5000))]),
        {"clf__C": [0.01, 0.1, 1],
         "clf__class_weight": [None, "balanced"]}),
    "Complement NB": (
        Pipeline([("features", make_features("combined", nonneg=True)), ("clf", ComplementNB())]),
        {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
    "Random forest": (
        Pipeline([("features", make_features("combined", ngram_range=(1, 1), max_features=10_000)),
                  ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=N_JOBS))]),
        {"clf__min_samples_leaf": [1, 3],
         "clf__class_weight": [None, "balanced_subsample"]}),
}

exp2_rows, fitted, exp2_scores, cv_tables = [], {}, {}, {}
for name, (pipe, grid) in searches.items():
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, scoring="average_precision", cv=CV, n_jobs=N_JOBS, refit=True)
    gs.fit(X_train, y_train)
    fitted[name] = gs.best_estimator_
    cv_tables[name] = pd.DataFrame(gs.cv_results_)
    s = get_scores(gs.best_estimator_, X_test); exp2_scores[name] = s
    exp2_rows.append({"model": name, "best params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()},
                      "CV PR-AUC": gs.best_score_,
                      **{f"Test {k}": v for k, v in evaluate(y_test, s).items()},
                      "minutes": round((time.time() - t0) / 60, 1)})
    print(f"{name:<20} CV PR-AUC={gs.best_score_:.3f}  best={gs.best_params_}  ({time.time()-t0:.0f}s)")

exp2 = pd.DataFrame(exp2_rows).set_index("model").sort_values("CV PR-AUC", ascending=False)
exp2.to_csv(RES_DIR / "exp2_model_comparison.csv")
pd.concat(cv_tables, names=["model"]).to_csv(RES_DIR / "exp2_all_grid_results.csv")
exp2.round(3)

# %%
# Final model = best CV PR-AUC (chosen on training data only, never on the test set)
BEST_NAME = exp2.index[0]
best_model = fitted[BEST_NAME]
best_scores_test = exp2_scores[BEST_NAME]
print(f"Selected model: {BEST_NAME}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
order = exp2.index[::-1]
axes[0].barh(order, exp2.loc[order, "CV PR-AUC"], color=[FRAUD_C if m == BEST_NAME else GREY_C for m in order])
axes[0].scatter(exp2.loc[order, "Test PR-AUC"], order, color="black", zorder=3, label="Test PR-AUC")
for i, m in enumerate(order):
    axes[0].text(exp2.loc[m, "CV PR-AUC"] + 0.01, i, f"{exp2.loc[m, 'CV PR-AUC']:.2f}", va="center")
axes[0].set_xlim(0, 1.05); axes[0].set_xlabel("PR-AUC (bars: tuned CV · dots: test)")
axes[0].set_title("Tuned model comparison"); axes[0].legend(loc="lower right")

# Hyperparameter sensitivity for the baseline: regularisation × imbalance handling
lr_cv = cv_tables["Logistic regression"].copy()
lr_cv["ngram"] = lr_cv[f"param_{tfidf_param('ngram_range')}"].astype(str)
lr_cv["cw"] = lr_cv["param_clf__class_weight"].apply(lambda v: "none" if v is None or v is np.ma.masked
                                                     or pd.isna(v) else str(v))
for (ng, cw), g in lr_cv.groupby(["ngram", "cw"], dropna=False):
    g = g.sort_values("param_clf__C")
    axes[1].plot(g["param_clf__C"].astype(float), g["mean_test_score"], marker="o",
                 ls="-" if cw == "balanced" else "--", label=f"n-grams {ng}, class weight {cw}")
axes[1].set_xscale("log"); axes[1].set_xlabel("C (inverse regularisation)"); axes[1].set_ylabel("CV PR-AUC")
axes[1].set_title("Logistic regression tuning"); axes[1].legend(fontsize=8)
savefig("05_exp2_model_comparison_and_tuning")

# %% [markdown]
# ## 8. Interpretation — what does fraud language look like?
# We use the tuned **logistic regression** (interpretable by design) regardless of which model won, because its
# coefficients show directly which words and metadata push a posting towards "fraudulent".

# %%
lr = fitted["Logistic regression"]
names = lr.named_steps["features"].get_feature_names_out()
coefs = lr.named_steps["clf"].coef_.ravel()

def pretty(n):
    return (n.replace("tfidf__", "").replace("flags__", "[meta] ").replace("cat__", "[meta] ")
             .replace("text_stats__", "[stat] "))

coef_df = pd.DataFrame({"feature": [pretty(n) for n in names], "coef": coefs})
coef_df.sort_values("coef").to_csv(RES_DIR / "lr_coefficients.csv", index=False)
top_pos = coef_df.nlargest(20, "coef")
top_neg = coef_df.nsmallest(20, "coef")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].barh(top_pos["feature"][::-1], top_pos["coef"][::-1], color=FRAUD_C)
axes[0].set_title("Strongest signals of FRAUD"); axes[0].set_xlabel("Coefficient")
axes[1].barh(top_neg["feature"][::-1], top_neg["coef"][::-1], color=REAL_C)
axes[1].set_title("Strongest signals of LEGITIMATE"); axes[1].set_xlabel("Coefficient")
savefig("06_interpretation_top_features")

# %% [markdown]
# ## 9. Unsupervised analysis — topic modeling with NMF
# Two questions, both answered **without using the fraud labels to build the topics**:
# * **9a.** Across *all* postings, which themes exist, and which themes concentrate fraud?
# * **9b.** Among fraudulent postings only, do distinct *fraud archetypes* emerge?
#
# We use Non-negative Matrix Factorisation (NMF) on TF-IDF, which typically gives more coherent topics than LDA on
# short business text. The number of topics *k* is tuned using reconstruction error (elbow) plus manual
# inspection of topic coherence.

# %%
def fit_topics(texts, k, max_features=5000):
    vec = TfidfVectorizer(min_df=5, max_df=0.7, stop_words="english", max_features=max_features,
                          sublinear_tf=True)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=400)
    W = nmf.fit_transform(X)
    return vec, nmf, W, X

def top_words(vec, nmf, n=8):
    vocab = np.array(vec.get_feature_names_out())
    return [", ".join(vocab[np.argsort(-row)[:n]]) for row in nmf.components_]

all_texts = train_text["text_all"]
fraud_texts = train_text.loc[train_text["fraudulent"] == 1, "text_all"]

# Tune k with reconstruction error
ks = [4, 6, 8, 10, 12, 15]
vec_tmp = TfidfVectorizer(min_df=5, max_df=0.7, stop_words="english", max_features=5000, sublinear_tf=True)
X_tmp = vec_tmp.fit_transform(all_texts)
errs = [NMF(n_components=k, init="nndsvda", random_state=RANDOM_STATE, max_iter=400).fit(X_tmp).reconstruction_err_
        for k in ks]
plt.figure(figsize=(5, 3)); plt.plot(ks, errs, marker="o", color=REAL_C)
plt.xlabel("Number of topics k"); plt.ylabel("Reconstruction error"); plt.title("Choosing k (all postings)")
savefig("07_nmf_choose_k")

# %%
K_ALL = 10   # adjust after inspecting the elbow plot and the topic words below
vec_all, nmf_all, W_all, _ = fit_topics(all_texts, K_ALL)
topic_tbl = pd.DataFrame({"top words": top_words(vec_all, nmf_all)})
topic_tbl["dominant topic share"] = pd.Series(W_all.argmax(1)).value_counts(normalize=True).sort_index()
topic_tbl["fraud rate"] = pd.Series(train_text["fraudulent"].values).groupby(W_all.argmax(1)).mean()
topic_tbl = topic_tbl.sort_values("fraud rate", ascending=False)
topic_tbl.to_csv(RES_DIR / "nmf_topics_all_postings.csv")
pd.set_option("display.max_colwidth", 120)
topic_tbl.round(3)

# %%
plt.figure(figsize=(9, 4.5))
lbl = [f"T{i}: " + ", ".join(w.split(", ")[:4]) for i, w in zip(topic_tbl.index, topic_tbl["top words"])]
plt.barh(lbl[::-1], topic_tbl["fraud rate"][::-1] * 100,
         color=[FRAUD_C if r > 2 * base else GREY_C for r in topic_tbl["fraud rate"][::-1]])
plt.axvline(base * 100, color="black", ls="--", lw=1)
plt.xlabel("% fraudulent among postings whose dominant topic is this one")
plt.title("Which themes concentrate fraud? (NMF topics, no labels used)")
savefig("08_nmf_fraud_rate_by_topic")

# %%
# 9b. Fraud archetypes — topics fitted on fraudulent postings only
K_FRAUD = 5
vec_f, nmf_f, W_f, _ = fit_topics(fraud_texts, K_FRAUD, max_features=3000)
fraud_sub = train_text.loc[fraud_texts.index].assign(archetype=W_f.argmax(1))
arch = pd.DataFrame({"top words": top_words(vec_f, nmf_f, 10),
                     "share of fraud": fraud_sub["archetype"].value_counts(normalize=True).sort_index(),
                     "example titles": fraud_sub.groupby("archetype")["title"]
                         .apply(lambda t: " | ".join(t.dropna().astype(str).value_counts().index[:3])),
                     "% no logo": fraud_sub.groupby("archetype")["has_company_logo"].apply(lambda s: 1 - s.mean()),
                     "% telecommute": fraud_sub.groupby("archetype")["telecommuting"].mean()})
arch.to_csv(RES_DIR / "nmf_fraud_archetypes.csv")
arch.round(2)

# %% [markdown]
# ## 10. Fraud-risk score & review-budget analysis
# Instead of a hard fake/real label, every posting gets a **risk score from 0 to 100**: its percentile among
# training-set scores (so 95 means "riskier than 95% of postings seen in training"). A platform can then route
# postings into **review tiers** and set the tier cut-offs to match moderator capacity.

# %%
train_scores = get_scores(best_model, X_train)
def risk_score(raw):
    return 100 * np.searchsorted(np.sort(train_scores), raw, side="right") / len(train_scores)

test_risk = risk_score(best_scores_test)

# Capture curve: share of all fraud caught vs share of postings reviewed
order_idx = np.argsort(-best_scores_test)
caught = np.cumsum(y_test[order_idx]) / y_test.sum()
reviewed = np.arange(1, len(y_test) + 1) / len(y_test)

budgets = [0.01, 0.02, 0.05, 0.10, 0.20]
budget_tbl = pd.DataFrame({
    "% of postings reviewed": [f"{b:.0%}" for b in budgets],
    "postings to review (per 10,000)": [int(b * 10_000) for b in budgets],
    "% of fraud caught": [recall_at_budget(y_test, best_scores_test, b) for b in budgets],
    "% of reviewed that are fraud": [precision_at_budget(y_test, best_scores_test, b) for b in budgets],
    "random review catches": budgets})
budget_tbl.to_csv(RES_DIR / "review_budget_table.csv", index=False)

plt.figure(figsize=(7, 4.5))
plt.plot(reviewed * 100, caught * 100, color=FRAUD_C, lw=2.5, label=f"{BEST_NAME} risk ranking")
plt.plot([0, 100], [0, 100], color=GREY_C, ls="--", label="Random review")
r5 = recall_at_budget(y_test, best_scores_test)
plt.scatter([REVIEW_BUDGET * 100], [r5 * 100], color="black", zorder=3)
plt.annotate(f"Review top {REVIEW_BUDGET:.0%} → catch {r5:.0%} of fraud",
             (REVIEW_BUDGET * 100, r5 * 100), xytext=(15, r5 * 100 - 18), arrowprops=dict(arrowstyle="->"))
plt.xlim(0, 40); plt.ylim(0, 102)
plt.xlabel("% of postings reviewed (highest risk first)"); plt.ylabel("% of all fraudulent postings caught")
plt.title("Moderator review budget vs fraud caught (test set)"); plt.legend(loc="lower right")
savefig("09_risk_score_capture_curve")
budget_tbl.round(3)

# %%
# Review tiers (cut-offs are a business choice; these are illustrative)
tiers = pd.cut(test_risk, bins=[-0.1, 80, 95, 100], labels=["Low (publish)", "Medium (spot-check)", "High (hold for review)"])
tier_tbl = pd.DataFrame({"tier": tiers, "fraud": y_test}).groupby("tier", observed=False)["fraud"].agg(
    postings="size", fraud_rate="mean", fraud_caught="sum")
tier_tbl["share of all fraud"] = tier_tbl["fraud_caught"] / y_test.sum()
tier_tbl.to_csv(RES_DIR / "risk_tiers.csv")
tier_tbl.round(3)

# %% [markdown]
# ## 11. Error analysis
# What does the model miss (false negatives) and what does it wrongly flag (false positives)?
# These cases guide future work and show moderators why human review stays essential.

# %%
test_view = data.loc[X_test.index, ["title", "location", "industry", "has_company_logo", "has_salary",
                                    "telecommuting", "fraudulent"]].assign(risk=test_risk.round(1))
print("Missed fraud (fraudulent postings with the LOWEST risk):")
display_fn = globals().get("display", print)
display_fn(test_view[test_view["fraudulent"] == 1].nsmallest(8, "risk"))
print("\nFalse alarms (legitimate postings with the HIGHEST risk):")
display_fn(test_view[test_view["fraudulent"] == 0].nlargest(8, "risk"))

# %%
plt.figure(figsize=(7, 3.5))
plt.hist(test_risk[y_test == 0], bins=40, alpha=0.6, color=REAL_C, label="Legitimate", density=True)
plt.hist(test_risk[y_test == 1], bins=40, alpha=0.6, color=FRAUD_C, label="Fraudulent", density=True)
plt.xlabel("Risk score (0–100)"); plt.ylabel("Density"); plt.title("Risk score distribution (test set)")
plt.legend()
savefig("10_risk_score_distribution")

# %% [markdown]
# ## 12. Summary of results
# Key numbers for the report and slides are saved to `results/summary.json`.

# %%
summary = {
    "n_postings": int(len(data)), "fraud_rate": round(float(base), 4),
    "exp1_cv_pr_auc": exp1["CV PR-AUC"].round(3).to_dict(),
    "exp1_test_pr_auc": exp1["Test PR-AUC"].round(3).to_dict(),
    "best_model": BEST_NAME,
    "best_model_params": {k: str(v) for k, v in exp2.loc[BEST_NAME, "best params"].items()},
    "best_model_test": {k.replace("Test ", ""): round(float(exp2.loc[BEST_NAME, k]), 3)
                        for k in exp2.columns if k.startswith("Test ")},
    "baseline_lr_test_pr_auc": round(float(exp2.loc["Logistic regression", "Test PR-AUC"]), 3),
    "review_budget": {r["% of postings reviewed"]: round(float(r["% of fraud caught"]), 3)
                      for _, r in budget_tbl.iterrows()},
    "top_fraud_signals": top_pos["feature"].head(10).tolist(),
}
(RES_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

# %% [markdown]
# ### Interpretation checklist (fill in after running)
# * **Signal source:** Does text beat metadata? Does combining add meaningfully over text alone?
# * **Complexity:** Is the best tuned model clearly better than the tuned logistic-regression baseline,
#   or within CV noise? If within noise, the interpretable model is the better business choice.
# * **Imbalance:** Did `class_weight="balanced"` help PR-AUC, or mainly shift the threshold?
# * **Fraud language:** Which themes dominate the top fraud coefficients (e.g. money, work-from-home, data entry,
#   missing company information, contact placeholders)?
# * **Archetypes:** Name each fraud archetype from 9b in plain business language.
# * **Operations:** At a 5% review budget, what share of fraud is caught, and how many postings per day would that
#   mean for a platform of a given size?
