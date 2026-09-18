# Hotel Booking Cancellation Prediction

## Project Overview

This project develops a machine learning system to predict whether a hotel booking is likely to be cancelled.

The project compares multiple classification algorithms and evaluates their performance using Accuracy, Precision, Recall and F1 Score. The final model is selected based on its F1 Score and deployed through an interactive Streamlit dashboard.

## Business Problem

Hotel booking cancellations create uncertainty in occupancy planning, revenue management and resource allocation.

Predicting cancellation risk in advance can help hotels identify potentially high-risk bookings and support proactive actions such as targeted reminders, confirmation communication and appropriate booking policies.

## Dataset

The project uses the Hotel Booking Demand dataset.

- Original records: 119,390
- Records after removing duplicates: 87,396
- Original features: 32
- Target variable: `is_canceled`

Target variable:

- `0` = Booking not cancelled
- `1` = Booking cancelled

## Data Preparation

The following preprocessing steps were performed:

- Duplicate records were removed.
- `company` was removed because of a high proportion of missing values.
- `reservation_status` and `reservation_status_date` were removed because they contain information about the eventual reservation outcome and could cause data leakage.
- Numerical variables were imputed using the median and standardized.
- Categorical variables were imputed using the most frequent value and one-hot encoded.
- The dataset was divided into training and testing sets using an 80:20 stratified split.

## Machine Learning Models

Nine classification models were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Naive Bayes
4. K-Nearest Neighbours (KNN)
5. Support Vector Machine (SVM)
6. Random Forest
7. AdaBoost
8. XGBoost
9. Artificial Neural Network (ANN)

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| XGBoost | 84.94% | 75.40% | 67.10% | 71.01% |
| Artificial Neural Network | 83.80% | 73.15% | 64.91% | 68.78% |
| Decision Tree | 80.97% | 67.57% | 59.19% | 63.10% |
| KNN | 78.91% | 62.82% | 57.00% | 59.77% |
| Logistic Regression | 79.43% | 67.67% | 48.18% | 56.29% |
| SVM | 79.37% | 68.57% | 46.08% | 55.12% |
| Naive Bayes | 37.87% | 30.52% | 98.75% | 46.63% |
| Random Forest | 79.22% | 85.27% | 29.51% | 43.85% |
| AdaBoost | 75.03% | 80.22% | 12.15% | 21.11% |

## Final Model

### XGBoost

XGBoost was selected as the final model because it achieved the highest F1 Score among the nine models evaluated.

Performance on the held-out test dataset:

- Accuracy: **84.94%**
- Precision: **75.40%**
- Recall: **67.10%**
- F1 Score: **71.01%**

F1 Score was used as the primary model-selection criterion because it provides a balance between precision and recall.

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard with five sections:

- **Overview** – Project objective, business problem and methodology
- **Data Insights** – Historical cancellation patterns
- **Model Comparison** – Performance comparison of all nine models
- **Prediction** – Interactive cancellation-risk prediction
- **Business Insights** – Managerial implications and limitations

## Business Value

A cancellation prediction system can support hotels in:

- Identifying potentially high-risk bookings
- Improving occupancy planning
- Supporting revenue management
- Prioritizing proactive customer communication
- Developing appropriate booking and confirmation strategies

The model is intended as a decision-support tool and should be considered alongside operational context and managerial judgement.

## Project Files

- `app.py` – Streamlit dashboard
- `train_model.py` – Model training and evaluation
- `model_comparison.csv` – Performance results of the nine models
- `best_model_name.txt` – Name of the selected model

## Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## How to Run

Install the required libraries:

```bash
pip install pandas scikit-learn xgboost joblib streamlit
