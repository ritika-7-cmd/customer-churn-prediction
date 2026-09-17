# Customer Churn Prediction - Telco Dataset

End-to-end Machine Learning project to predict customer churn for a telecommunications company using the IBM Telco Customer Churn dataset.

## Project Overview

- **Problem**: Binary Classification (Predict whether a customer will churn or not)
- **Dataset**: IBM Telco Customer Churn (7,043 customers, 21 features)
- **Best Model**: Logistic Regression
- **Test ROC-AUC**: 0.845
- **Recall (Churn class)**: 0.78

## Key Insights

- Month-to-month contract customers have the highest churn rate
- Customers with low tenure (especially < 12 months) are more likely to churn
- Fiber optic internet service and Electronic check payment method are associated with higher churn
- Customers with fewer additional services are at higher risk

## Project Structure

```
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned data
├── src/                     # Main Python script
├── models/                  # Saved trained model
└── reports/
    ├── figures/             # All visualizations
    └── evaluation files
```

## How to Run

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python src/churn_prediction_project.py
   ```

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

## Author

Your Name
