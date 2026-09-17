#!/usr/bin/env python3
"""
=============================================================
END-TO-END CUSTOMER CHURN PREDICTION PROJECT
Telco Customer Churn Dataset (IBM Sample)
=============================================================
This script performs a complete data analysis + ML pipeline:
1. Problem Definition
2. Data Loading
3. Exploratory Data Analysis (EDA)
4. Data Cleaning & Preprocessing
5. Feature Engineering
6. Model Training & Comparison
7. Model Evaluation
8. Feature Importance & Insights
9. Model Persistence
10. Business Recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

OUTPUT_DIR = "churn_project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("STEP 1: PROBLEM DEFINITION & BUSINESS UNDERSTANDING")
print("=" * 80)
print("""
Business Context:
- A telecommunications company is losing customers (churn).
- Acquiring a new customer costs 5-25x more than retaining an existing one.
- Goal: Predict which customers are likely to churn so the company can
  proactively offer retention incentives (discounts, better plans, support).

Problem Type: Binary Classification (Churn = Yes/No)

Success Metrics (Business-aligned):
- High Recall on Churn class (we don't want to miss potential churners)
- Good Precision (don't waste retention budget on loyal customers)
- ROC-AUC > 0.80 is typically considered strong for this problem
- Actionable insights on WHY customers churn

Dataset: IBM Telco Customer Churn (7043 customers, 21 features)
""")

print("\n" + "=" * 80)
print("STEP 2: DATA LOADING")
print("=" * 80)

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic info:")
df.info()

# Save raw data snapshot
df.to_csv(f"{OUTPUT_DIR}/01_raw_data.csv", index=False)
print(f"\n[Saved] Raw data snapshot → {OUTPUT_DIR}/01_raw_data.csv")

print("\n" + "=" * 80)
print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# 3.1 Target distribution
print("\n--- 3.1 Target Variable Distribution ---")
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100
print(churn_counts)
print(f"\nChurn Rate: {churn_pct['Yes']:.2f}%")
print(f"Retention Rate: {churn_pct['No']:.2f}%")
print("→ Mild class imbalance (~26.5% churn). We will handle this carefully.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
churn_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Churn Count')
axes[0].set_ylabel('Number of Customers')
axes[0].set_xticklabels(['No', 'Yes'], rotation=0)
axes[1].pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Churn Proportion')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_target_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] Target distribution plot → {OUTPUT_DIR}/02_target_distribution.png")

# 3.2 Missing values
print("\n--- 3.2 Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No NaN values detected.")
# TotalCharges is often stored as object with blank strings
print(f"\nTotalCharges unique blank-like values: {(df['TotalCharges'].astype(str).str.strip() == '').sum()}")

# 3.3 Numerical features summary
print("\n--- 3.3 Numerical Features Summary ---")
print(df.describe())

# 3.4 Categorical features
print("\n--- 3.4 Categorical Features Unique Values ---")
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique → {df[col].unique()[:10]}")

# 3.5 Key visualizations
print("\n--- 3.5 Creating EDA Visualizations ---")

# Tenure vs Churn
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True, ax=ax,
             palette={'No': '#2ecc71', 'Yes': '#e74c3c'})
ax.set_title('Tenure Distribution by Churn Status')
ax.set_xlabel('Tenure (months)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_tenure_vs_churn.png", dpi=150, bbox_inches='tight')
plt.close()

# MonthlyCharges vs Churn
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=30, kde=True, ax=ax,
             palette={'No': '#2ecc71', 'Yes': '#e74c3c'})
ax.set_title('Monthly Charges Distribution by Churn Status')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_monthly_charges_vs_churn.png", dpi=150, bbox_inches='tight')
plt.close()

# Contract type vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
contract_churn.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'])
ax.set_title('Churn Rate by Contract Type')
ax.set_ylabel('Percentage')
ax.set_xlabel('Contract Type')
ax.legend(title='Churn')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_contract_vs_churn.png", dpi=150, bbox_inches='tight')
plt.close()

# Internet Service vs Churn
fig, ax = plt.subplots(figsize=(8, 5))
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
internet_churn.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'])
ax.set_title('Churn Rate by Internet Service')
ax.set_ylabel('Percentage')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_internet_vs_churn.png", dpi=150, bbox_inches='tight')
plt.close()

# Payment Method vs Churn
fig, ax = plt.subplots(figsize=(10, 5))
pay_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index') * 100
pay_churn.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'])
ax.set_title('Churn Rate by Payment Method')
ax.set_ylabel('Percentage')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_payment_vs_churn.png", dpi=150, bbox_inches='tight')
plt.close()

print("[Saved] Multiple EDA plots (tenure, charges, contract, internet, payment)")

# Correlation for numerical
print("\n--- 3.6 Correlation of Numerical Features ---")
df_temp = df.copy()
df_temp['TotalCharges'] = pd.to_numeric(df_temp['TotalCharges'], errors='coerce')
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
corr = df_temp[num_cols].corr()
print(corr)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Heatmap (Numerical Features)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] Correlation heatmap → {OUTPUT_DIR}/08_correlation_heatmap.png")

print("\n" + "=" * 80)
print("STEP 4: DATA CLEANING & PREPROCESSING")
print("=" * 80)

df_clean = df.copy()

# 4.1 Fix TotalCharges
print("\n--- 4.1 Handling TotalCharges ---")
# Blank strings exist for customers with tenure=0
blank_mask = df_clean['TotalCharges'].astype(str).str.strip() == ''
print(f"Blank TotalCharges rows: {blank_mask.sum()}")
print(df_clean.loc[blank_mask, ['customerID', 'tenure', 'TotalCharges', 'Churn']])

df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
# Impute with 0 (tenure=0 → no charges yet) or median; 0 is more logical
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
print("Filled blank TotalCharges with 0 (new customers).")

# 4.2 Drop customerID (identifier, not predictive)
print("\n--- 4.2 Dropping customerID ---")
df_clean.drop('customerID', axis=1, inplace=True)
print("Dropped customerID.")

# 4.3 Encode target
print("\n--- 4.3 Encoding Target Variable ---")
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
print(f"Churn mapped: Yes→1, No→0. Value counts:\n{df_clean['Churn'].value_counts()}")

# 4.4 Check for any remaining issues
print("\n--- 4.4 Final Missing Value Check ---")
print(df_clean.isnull().sum().sum(), "missing values remaining.")
print(f"Cleaned shape: {df_clean.shape}")

df_clean.to_csv(f"{OUTPUT_DIR}/09_cleaned_data.csv", index=False)
print(f"[Saved] Cleaned data → {OUTPUT_DIR}/09_cleaned_data.csv")

print("\n" + "=" * 80)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 80)

df_fe = df_clean.copy()

# 5.1 Create new features
print("\n--- 5.1 Creating Derived Features ---")

# Average monthly charge consistency
df_fe['AvgMonthlyCharge'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)  # +1 to avoid div0
print("Created: AvgMonthlyCharge = TotalCharges / (tenure + 1)")

# Tenure groups (customer lifecycle stage)
df_fe['TenureGroup'] = pd.cut(df_fe['tenure'],
                              bins=[-1, 12, 24, 48, 72],
                              labels=['0-12m', '13-24m', '25-48m', '49-72m'])
print("Created: TenureGroup (0-12, 13-24, 25-48, 49-72 months)")

# Has both partner and dependents (family indicator)
df_fe['IsFamily'] = ((df_fe['Partner'] == 'Yes') & (df_fe['Dependents'] == 'Yes')).astype(int)
print("Created: IsFamily (Partner=Yes AND Dependents=Yes)")

# Number of additional services (proxy for engagement)
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df_fe['NumAdditionalServices'] = 0
for col in service_cols:
    df_fe['NumAdditionalServices'] += (df_fe[col] == 'Yes').astype(int)
print("Created: NumAdditionalServices (count of Yes in 6 service columns)")

# High monthly charge flag
df_fe['HighMonthlyCharge'] = (df_fe['MonthlyCharges'] > df_fe['MonthlyCharges'].median()).astype(int)
print("Created: HighMonthlyCharge (above median)")

print(f"\nNew feature set shape: {df_fe.shape}")
print(f"New columns: {[c for c in df_fe.columns if c not in df_clean.columns]}")

print("\n" + "=" * 80)
print("STEP 6: TRAIN-TEST SPLIT & PREPROCESSING PIPELINE")
print("=" * 80)

# Separate features and target
X = df_fe.drop('Churn', axis=1)
y = df_fe['Churn']

# Identify column types
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
                    'AvgMonthlyCharge', 'IsFamily', 'NumAdditionalServices', 'HighMonthlyCharge']
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Stratified split to preserve churn rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples (Churn rate: {y_train.mean()*100:.2f}%)")
print(f"Test set:  {X_test.shape[0]} samples (Churn rate: {y_test.mean()*100:.2f}%)")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

print("\nPreprocessing pipeline created:")
print("  - Numeric → StandardScaler")
print("  - Categorical → OneHotEncoder (handle_unknown='ignore')")

print("\n" + "=" * 80)
print("STEP 7: MODEL TRAINING & COMPARISON")
print("=" * 80)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=8),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42,
                                            class_weight='balanced', max_depth=12, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, random_state=42,
                                                    max_depth=5, learning_rate=0.1)
}

results = []
trained_pipelines = {}

print("\nTraining models with 5-fold Stratified Cross-Validation...\n")

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Fit on full training data
    pipe.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    
    results.append({
        'Model': name,
        'CV_ROC_AUC_mean': cv_scores.mean(),
        'CV_ROC_AUC_std': cv_scores.std(),
        'Test_Accuracy': acc,
        'Test_Precision': prec,
        'Test_Recall': rec,
        'Test_F1': f1,
        'Test_ROC_AUC': auc,
        'Test_AvgPrecision': ap
    })
    
    trained_pipelines[name] = pipe
    print(f"{name:25s} | CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
          f"Test AUC: {auc:.4f} | F1: {f1:.4f} | Recall: {rec:.4f}")

results_df = pd.DataFrame(results).sort_values('Test_ROC_AUC', ascending=False)
print("\n=== Model Comparison Table ===")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
results_df.to_csv(f"{OUTPUT_DIR}/10_model_comparison.csv", index=False)
print(f"[Saved] Model comparison → {OUTPUT_DIR}/10_model_comparison.csv")

# Select best model by Test ROC-AUC
best_model_name = results_df.iloc[0]['Model']
best_pipe = trained_pipelines[best_model_name]
print(f"\n★ Best Model Selected: {best_model_name}")

print("\n" + "=" * 80)
print("STEP 8: DETAILED EVALUATION OF BEST MODEL")
print("=" * 80)

y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
ax.set_title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] Confusion matrix → {OUTPUT_DIR}/11_confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC (AUC = {roc_auc_score(y_test, y_prob):.4f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curve - {best_model_name}')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] ROC curve → {OUTPUT_DIR}/12_roc_curve.png")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, color='#e74c3c', lw=2,
        label=f'PR (AP = {average_precision_score(y_test, y_prob):.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Precision-Recall Curve - {best_model_name}')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_pr_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved] PR curve → {OUTPUT_DIR}/13_pr_curve.png")

print("\n" + "=" * 80)
print("STEP 9: FEATURE IMPORTANCE & BUSINESS INSIGHTS")
print("=" * 80)

# Extract feature names after one-hot
ohe = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
ohe_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + ohe_feature_names

# Get importances depending on model type
classifier = best_pipe.named_steps['classifier']

if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
elif hasattr(classifier, 'coef_'):
    importances = np.abs(classifier.coef_[0])
else:
    importances = None

if importances is not None:
    feat_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv(f"{OUTPUT_DIR}/14_feature_importance.csv", index=False)
    
    # Plot top 15
    fig, ax = plt.subplots(figsize=(10, 7))
    top15 = feat_imp.head(15)
    sns.barplot(data=top15, y='Feature', x='Importance', ax=ax, palette='viridis')
    ax.set_title(f'Top 15 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/15_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Feature importance plot & CSV")
else:
    print("Model does not expose feature importances directly.")

print("\n" + "=" * 80)
print("STEP 10: MODEL PERSISTENCE")
print("=" * 80)

model_path = f"{OUTPUT_DIR}/best_churn_model.joblib"
joblib.dump(best_pipe, model_path)
print(f"Best model pipeline saved to: {model_path}")

# Also save a simple prediction function example
print("""
How to load and use the model later:
------------------------------------
import joblib
import pandas as pd

model = joblib.load('churn_project_outputs/best_churn_model.joblib')
# new_data must have the same columns as X (after feature engineering)
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
""")

print("\n" + "=" * 80)
print("STEP 11: BUSINESS RECOMMENDATIONS & SUMMARY")
print("=" * 80)

print(f"""
PROJECT SUMMARY
===============
Dataset          : IBM Telco Customer Churn (7043 customers)
Best Model       : {best_model_name}
Test ROC-AUC     : {results_df.iloc[0]['Test_ROC_AUC']:.4f}
Test F1-Score    : {results_df.iloc[0]['Test_F1']:.4f}
Test Recall      : {results_df.iloc[0]['Test_Recall']:.4f}  (ability to catch churners)
Test Precision   : {results_df.iloc[0]['Test_Precision']:.4f}

KEY DRIVERS OF CHURN (from EDA + Feature Importance):
1. Contract Type          → Month-to-month customers churn far more than yearly contracts
2. Tenure                 → New customers (especially < 12 months) are highest risk
3. Internet Service       → Fiber optic customers show elevated churn (possibly price/quality)
4. Payment Method         → Electronic check users churn more than automatic payments
5. Monthly Charges        → Higher charges correlate with higher churn probability
6. Lack of add-on services→ Customers with fewer security/support services leave more

ACTIONABLE RECOMMENDATIONS:
1. Target month-to-month customers with tenure < 12 months for retention offers.
2. Incentivize switching to 1-year or 2-year contracts (discounts, free upgrades).
3. Improve onboarding experience for new customers in first 3-6 months.
4. Encourage automatic payment methods (bank transfer / credit card).
5. Bundle online security, tech support, and backup for at-risk customers.
6. Review fiber optic pricing/quality – high churn despite higher revenue.
7. Deploy the model in production to score customers monthly and trigger
   personalized retention campaigns for those with high churn probability.

FILES GENERATED:
- All plots, cleaned data, model comparison, feature importance, and the
  trained model pipeline are saved inside the folder: {OUTPUT_DIR}/
""")

print("\n" + "=" * 80)
print("END-TO-END PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)
