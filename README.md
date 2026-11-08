# EvalSphere

**Benchmark Smarter. Decide Faster.**

**Student:** SRIRAM POLAMARASETTI  
**BITS ID:** 2026ak05608

EvalSphere is a machine-learning classification and benchmarking workspace. It
compares five classifiers, explores dataset characteristics, evaluates
predictions, and presents the Mobile Price Classification assignment results.

## Project links

- **Repository URL:** [Open the GitHub repository](https://github.com/sreeram348/2026ak05608_assignment_2.git)
- **Live app URL:** [Open EvalSphere](https://2026ak05608assignment2git-tgcyacyp3j2ftfuwllshbx.streamlit.app/)

## Dataset

The assignment uses the Mobile Price Classification dataset.

| Property | Value |
|---|---|
| Task | Multiclass classification |
| Records | 2,000 |
| Input features | 20 numeric mobile specifications |
| Target | `price_range` (four classes: 0–3) |
| Class distribution | 500 records per class |
| Missing values | None |

The dataset exceeds the assignment requirements of 500 records and 12 input
features. A stratified 80/20 split with random seed 42 is used. Preprocessing
is fitted on the training partition only. The 400-row reference holdout and a
balanced 100-row sample are included for evaluation.

## Models and results

The five classifiers are evaluated on the same holdout. Precision, recall, and
F1 use weighted averaging; AUC uses weighted one-vs-rest.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| K-Nearest Neighbors | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Gaussian Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Support Vector Machine | 0.8950 | 0.9580 | 0.8969 | 0.8950 | 0.8956 | 0.8602 |

Logistic Regression is the top performer on this fixed holdout, with 96.50%
accuracy and the highest AUC, F1, and MCC. Results describe this evaluation
split and do not guarantee performance on other data.

## Using the application

The app is organized into seven workspaces:

| Workspace | Purpose |
|---|---|
| Overview | Review the selected dataset and benchmark summary. |
| Dataset Hub | Search and load a sample dataset or upload a labeled CSV. |
| Benchmark Lab | Run and compare the five classifiers with interactive, labeled charts. |
| Feature Studio | Explore feature relevance, distributions, and correlations. |
| Prediction Center | Review holdout predictions and confidence, then export results. |
| Reports | Preview dataset and benchmark summaries and export CSV or PDF reports. |
| Assignment | View dataset verification, saved-model comparisons, and the confusion matrix and classification report together. |

### Run a benchmark

1. Open **Dataset Hub** and load a sample or upload a labeled classification CSV.
2. Select the target column and the holdout size.
3. Open **Benchmark Lab** and choose **Run benchmark**.
4. Compare the champion model, metric cards, leaderboard, and performance charts.
5. Explore features, review predictions, or export a report from the other workspaces.

### Review the assignment results

Open **Assignment** to see the Mobile Price dataset check, saved-model
comparison, and confusion matrix for the bundled holdout. Select a model to
review its classification report. Upload a matching labeled test CSV to
evaluate the saved models without retraining.

## Run locally

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`train_models.py` trains the saved models, creates the test CSV files, and
updates `results.csv`.
