Practical 1: Data Preprocessing and Exploration-------------------
This block combines the SQL queries from 1A and 1B with the comprehensive file-based EDA and feature engineering from 1C and 1D.

Python

# 1. Imports
import sqlite3, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
import warnings; warnings.filterwarnings("ignore")

# 2. Data Loading from SQL (from 1A & 1B)
def query_db(db_path, query, description):
    print(f"--- {description} ---")
    with sqlite3.connect(db_path) as con:
        print(pd.read_sql(query, con), "\n")

query_db('data/classic_rock.db', 
         'SELECT Artist, COUNT(*) AS num_songs FROM rock_songs GROUP BY 1 ORDER BY 2 DESC LIMIT 5;', 
         "1A: Top Rock Artists")
query_db('data/baseball.db', 
         'SELECT playerID, SUM(GP) AS num_games FROM allstarfull GROUP BY 1 ORDER BY 2 DESC LIMIT 3;', 
         "1B: Top Baseball Players")

# 3. EDA and Feature Engineering on Ames Housing Data (from 1C & 1D)
df = pd.read_csv("E:/SEM3/Practicals/ML/data/Ames_Housing_Data.tsv", sep='\t')
df = df.loc[df['Gr Liv Area'] <= 4000].copy() # Remove outliers

# --- Analysis & Cleaning (1C) --------------------------------------------------------------------------------------
print("--- 1C: Ames Housing Data Initial Analysis ---")
print(df['SalePrice'].describe())
print(f"\nMissing 'Lot Frontage' before imputation: {df['Lot Frontage'].isnull().sum()}")
df["Lot Frontage"].fillna(df["Lot Frontage"].median(), inplace=True)
print(f"Missing 'Lot Frontage' after imputation: {df['Lot Frontage'].isnull().sum()}")

# --- Transformation (1D) ----------------------------------------------------------------------------
df['SalePrice_log'] = np.log(df['SalePrice'])
skewed_cols = df.select_dtypes('number').skew().abs().sort_values(ascending=False)
skewed_cols = skewed_cols[skewed_cols > 0.75].index.drop(["SalePrice", "SalePrice_log"], errors='ignore')
df[skewed_cols] = df[skewed_cols].apply(np.log1p)
print(f"\nCorrected skew for {len(skewed_cols)} numeric columns.")

# --- Feature Engineering (1D) ---
print("\n--- 1D: Feature Engineering ---")
df['OQ_sq'] = df['Overall Qual']**2
df['OQ_x_YB'] = df['Overall Qual'] * df['Year Built']
print("New interaction features ('OQ_sq', 'OQ_x_YB'):\n", df[['OQ_sq', 'OQ_x_YB']].head())

pf = PolynomialFeatures(degree=2, include_bias=False)
poly_df = pd.DataFrame(pf.fit_transform(df[['Lot Area', 'Overall Qual']]), 
                       columns=pf.get_feature_names_out(['Lot Area', 'Overall Qual']))
print("\nGenerated Polynomial Features:\n", poly_df.head())
-----------------------------Practical 4: Discriminative Models-------------------------------------------------------------------------
This section combines the classification models from 4A (Logistic Regression), 4B (K-Nearest Neighbors), and 4C (Decision Tree).


# 1. Imports------------------------------------------
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# 2. Logistic Regression (from 4A)-----------------------------------------------------
print("--- 4A: Logistic Regression ---")
pima = pd.read_csv("data/pima-indians-diabetes.csv", header=None, names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'])
X_pima, y_pima = pima.drop('label', axis=1), pima.label
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_pima, y_pima, test_size=0.25, random_state=16)
log_model = LogisticRegression(max_iter=200, random_state=100).fit(X_train_p, y_train_p)
print(metrics.classification_report(y_test_p, log_model.predict(X_test_p)))

# 3. K-Nearest Neighbors (from 4B) & Decision Tree (from 4C)--------------------------------------------------------
print("\n--- 4B & 4C: KNN and Decision Tree ---")
X_tumor, y_tumor = pd.read_csv("data/tumor.csv", return_X_y=True)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tumor, y_tumor, test_size=0.2, stratify=y_tumor, random_state=123)

# --- KNN ---=============================================================----
f1_scores = [(k, metrics.f1_score(y_test_t, KNeighborsClassifier(n_neighbors=k).fit(X_train_t, y_train_t).predict(X_test_t))) for k in range(1, 11)]
best_k = sorted(f1_scores, key=lambda x: x[1], reverse=True)[0]
print(f"Best K for KNN is {best_k[0]} with F1-Score: {best_k[1]:.4f}")

# --- Decision Tree ---=======================================================
params = {'max_depth': [5, 10], 'min_samples_leaf': [1, 5], 'criterion': ['gini', 'entropy']}
grid = GridSearchCV(DecisionTreeClassifier(random_state=123), params, scoring='f1', cv=3).fit(X_train_t, y_train_t)
print("\nDecision Tree Best Params:", grid.best_params_)
print(f"Decision Tree F1 Score: {metrics.f1_score(y_test_t, grid.best_estimator_.predict(X_test_t)):.4f}")
Practicals 5 & 6: Generative & Probabilistic Models
This block covers Aim 5: Generative Models (Gaussian Naive Bayes) and Aim 6: Probabilistic Models (Bayesian Regression, GMM).

Python

# 1. Imports
import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# --- 5A: Gaussian Naive Bayes ---
print("--- 5A: Gaussian Naive Bayes ---")
X, y = make_classification(n_samples=800, n_features=6, n_informative=2, n_classes=3, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)
y_pred = GaussianNB().fit(x_train, y_train).predict(x_test)
print(f"Accuracy: {accuracy_score(y_pred, y_test):.4f}")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred); plt.title("GNB Confusion Matrix"); plt.show()

# --- 6A: Bayesian Linear Regression ---
print("\n--- 6A: Bayesian Linear Regression ---")
np.random.seed(42)
X_pts=np.linspace(-5, 5, 20); y_pts=2*X_pts+1+np.random.normal(0,2,X_pts.shape)
X_design = np.c_[np.ones_like(X_pts), X_pts]
prior_cov, noise_var = np.eye(2)*10, 4
post_cov = np.linalg.inv(np.linalg.inv(prior_cov) + (X_design.T@X_design)/noise_var)
post_mean = post_cov@(np.linalg.inv(prior_cov)@np.zeros(2) + (X_design.T@y_pts)/noise_var)
print("Posterior Mean (Intercept, Slope):", np.round(post_mean, 2))

# --- 6B: Gaussian Mixture Model ---
print("\n--- 6B: Gaussian Mixture Model ---")
X_iris, y_iris = load_iris(return_X_y=True)
gm = GaussianMixture(n_components=3, n_init=10, random_state=42).fit(X_iris)
y_pred_clusters = gm.predict(X_iris)
mapping = {stats.mode(y_pred_clusters[y_iris==cid]).mode[0]: cid for cid in np.unique(y_iris)}
y_pred_iris = np.array([mapping[cid] for cid in y_pred_clusters])
print(f"GMM Clustering Accuracy: {np.mean(y_pred_iris==y_iris):.4f}")
Practical 7: Model Evaluation-----------------------------------------------------------------------------
This final section covers Aim: Model Evaluation and Hyperparameter Tuning, focusing on the model evaluation part from 7A.

Python

# 1. Imports
from statistics import mean, stdev
import numpy as np
from sklearn import preprocessing, datasets, linear_model
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 2. Code for 7A: Stratified K-Fold Cross-Validation
X, y = datasets.load_breast_cancer(return_X_y=True)
x_scaled = preprocessing.MinMaxScaler().fit_transform(X)
lr = linear_model.LogisticRegression()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# 3. Perform Cross-Validation and Report Scores
scores = cross_val_score(lr, x_scaled, y, cv=skf)
print("--- 7A: Stratified K-Fold Cross-Validation ---")
print(f"Fold Accuracies: {np.round(scores, 3)}")
print(f"Mean Accuracy: {mean(scores)*100:.2f}% (Std Dev: {stdev(scores):.3f})")
