# Car Price Prediction

A machine learning web application that predicts the resale price of used cars using Decision Tree Regression.

## Features
- Predicts used car selling prices.
- Uses a tuned Decision Tree Regressor.
- Hyperparameter optimization with RandomizedSearchCV.
- Interactive Streamlit interface.
- Displays predicted prices in Indian Rupees (INR).

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Pickle

## Model Performance

| Metric | Score |
|---|---:|
| MAE | 0.6774 |
| MSE | 1.1187 |
| RMSE | 1.0577 |
| R² Score | 0.9514 |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model File
Place `car_price_model.pkl` in the project root alongside `app.py`.

## Disclaimer
Predictions are estimates based on the training dataset and should not be treated as guaranteed market prices.
