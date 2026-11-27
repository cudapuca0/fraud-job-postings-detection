# Project Status & Handoff

> **For teammates and their AI assistants:** read this file first. It records what exists, what the results are,
> what decisions were made, and what is left to do. Last updated: 2026-11-27.

## 1. The assignment (INSY 669/670 group project)
Course brief: "Social Media Management for Retail": apply text / image / network analytics to a real business problem.
Our topic (approved proposal): **Detecting Fraudulent Job Postings Using Text Analytics**.

**Five required deliverables**

1. Data file(s) + short data description (variables, source, sample size, cleaning)
2. Prompt log: every AI prompt, a summary of its output, and how it contributed
3. Code notebook (.ipynb/.py): sections for import, preprocessing, analysis, visualization, interpretation
4. Written report, **max 5 pages**: Executive Summary → Problem → Methodology → Key Results → Managerial Implications → Limitations & Future Directions
5. Presentation, **8–10 slides, 10 minutes max**, executive audience

**Grading:** novelty of questions 30% · appropriateness/reasoning of course techniques 30% ·
quality of tools + experimentation + **hyperparameter tuning** 30% · proposal 10% (done).

## 2. What has been done
* **Repo set up** (this repo, private for now; plan to make it public for portfolios after grading, with everyone's OK).
* **Full analysis notebook** `fraud_job_postings_analysis.ipynb` (same code in `fraud_job_postings_analysis.py`), run on the real data:
  1. Data import + description table (`results/data_description.csv`)
  2. Text cleaning: EMSCAD's hashed `#URL_…#`, `#EMAIL_…#`, `#PHONE_…#` placeholders → tokens; glued words split
  3. Feature groups: **TEXT** (TF-IDF over title/company profile/description/requirements/benefits + text stats) and
     **META** (logo, questions, telecommuting, salary given, missing-field flags, employment type, experience, education, industry, function, country)
  4. EDA charts (class balance, fraud rate by metadata and industry, profile length)
  5. **Experiment 1:** text vs metadata vs combined (same logistic regression)
  6. **Experiment 2:** 4 models tuned with GridSearchCV (5-fold, PR-AUC): logistic regression, linear SVM, Complement NB, random forest.
     Grids cover n-grams, regularisation C, class weighting, NB alpha, RF leaf size
  7. Interpretation: top logistic regression coefficients
  8. Unsupervised: NMF topic modeling (k chosen by elbow): fraud rate per topic and "fraud archetypes"
  9. **Fraud-risk score (0–100)**, capture curve (fraud caught vs % reviewed), review tiers
  10. Error analysis (missed fraud, false alarms)
* Outputs: 10 charts in `figures/`, result tables in `results/`, headline numbers in `results/summary.json`.
* Evaluation protocol: stratified 80/20 split, test set used once; all tuning by 5-fold CV on train.
  **Primary metric PR-AUC** (accuracy is useless: 95.2% of postings are legitimate). Also ROC-AUC, Recall@5% and Precision@5%.

## 3. Results so far (test set: 3,576 postings, 4.8% fraud)
| Finding | Number |
|---|---|
| Dataset | 17,880 postings, 866 fraudulent (4.8%) |
| Metadata only (LR) | PR-AUC 0.44 |
| Text only (LR) | PR-AUC 0.93 |
| Text + metadata (LR) | PR-AUC 0.92 (combining adds nothing) |
| Best tuned model: **Linear SVM** (C=1, no class weight) | PR-AUC **0.97**, ROC-AUC 0.99 |
| Tuned logistic regression (C=10, balanced, 1–2-grams) | PR-AUC 0.96 (close to the best and interpretable) |
| Random forest / Complement NB | PR-AUC 0.95 / 0.92 |
| **Review top 5% riskiest postings** | **catch 93% of fraud**; ~90% of reviewed postings are fraud |
| High-risk tier (225 of 3,576 postings) | holds 96% of all fraud |

**Fraud archetypes found by NMF (no labels used):** home-based data entry / payroll typist ("earn $100–200 daily"),
urgent part-time cash jobs, cruise-staff / visa / "free accommodation" offers, generic admin/customer-service postings.
Themes with the most fraud: home/sales-rep and oil & energy (Tidewater-style) postings; teaching abroad (TEFL) has almost none.

## 4. ⚠️ Known issue: fix before trusting the numbers
Some top fraud features are **company names** (`accion`, `aptitude staffing`, `link urltoken`).
The same scam operations post many times, so the model partly **memorizes companies** that appear in both train and test,
which inflates scores. **Fix:** group-aware split (`GroupShuffleSplit` / `StratifiedGroupKFold`), grouping on
company_profile text (or a hash of it). Postings without a profile need their own grouping, e.g. a hash of the description.
Rerun both experiments and report **both** the random-split and group-split scores. The drop is itself a finding worth
discussing under Limitations, and catching it strengthens the "thoroughness" grade.

## 5. To do
| # | Task | Notes |
|---|---|---|
| 1 | **Group-aware split fix** (section 4) | Highest priority; all numbers in the report depend on it |
| 2 | Review the notebook as a team | Everyone must be able to explain every section in Q&A |
| 3 | Name the NMF topics/archetypes in plain business language | `results/nmf_fraud_archetypes.csv`; consider k values other than 10/5 |
| 4 | Optional extra experiments | char n-grams, field-level TF-IDF (separate vectorizer per field), threshold choice for tiers |
| 5 | Data description write-up (deliverable 1) | Start from `results/data_description.csv`; dataset itself stays out of git (Kaggle link in README) |
| 6 | Keep the prompt log updated (deliverable 2) | `docs/prompt_log.md`: **every teammate adds their own prompts** |
| 7 | Write the 5-page report (deliverable 4) | Google Drive; structure in section 1 |
| 8 | Build the 8–10 slides (deliverable 5) | Outline in section 6 |
| 9 | Frame the topic for the course | The brief is social media and retail; frame it as platform trust & safety / brand impersonation on job platforms (LinkedIn, Indeed) |
| 10 | Before going public | Team consent, recruiter-friendly README with the capture-curve chart on top |

Owners and deadlines: **TBD, agree in the group chat.**

## 6. Slide outline (10 slides, ~1 min each)
1. Title + hook (harm of job scams to students and newcomers)
2. Problem: moderators can't check every posting, so this is a triage problem
3. Data: EMSCAD, 17,880 postings, 4.8% fraud (imbalance)
4. Approach: pipeline diagram (text / meta / combined → models → risk score; NMF side branch)
5. Where the signal is: text 0.93 vs metadata 0.44 (`figures/04_…`)
6. Experimentation & tuning (`figures/05_…`), the 30% tuning criterion
7. What fraud language looks like (`figures/06_…`)
8. Fraud archetypes from NMF (`figures/08_…` + archetype table)
9. **Risk score in practice: top 5% → 93% of fraud** (`figures/09_…`)
10. Recommendations (score routes postings to human review, not auto-removal) + limitations (company memorization, 2012–2014 data, scammer drift)

## 7. Working conventions
* Dataset: download `fake_job_postings.csv` from Kaggle ("Real or Fake Job Posting Prediction") into `data/` (gitignored).
* `pip install -r requirements.txt`; a full run takes about 15–25 min (the random forest grid is the slowest part).
* One person owns the main notebook; others experiment in `notebooks/<name>_*.ipynb` and merge what works.
* Clear notebook outputs before committing, or edit the `.py` version.
* Report, slides and prompt log drafts: shared Google Drive. Code, figures, results: this repo.
