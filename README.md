# 🚗 Car Price Prediction

A Machine Learning web application that predicts the resale value of used cars based on vehicle specifications. Built using **Python, Scikit-learn, and Streamlit**, with a tuned Decision Tree Regression model.

## 📌 Project Overview

The application estimates the selling price of a used car using information such as its manufacturing year, present market price, kilometres driven, fuel type, seller type, transmission, and ownership history.

The model uses **Decision Tree Regression**, with hyperparameter optimization performed using `RandomizedSearchCV`.

## ✨ Features

- Predicts the estimated resale price of used cars.
- Interactive web interface built with Streamlit.
- Handles categorical features using saved LabelEncoders.
- Uses a tuned Decision Tree Regression model.
- Displays predicted prices in Indian Rupees (₹).
- Includes model performance evaluation.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- RandomizedSearchCV

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Car_Name | Name of the car |
| Year | Manufacturing year |
| Present_Price | Current showroom price |
| Kms_Driven | Total kilometres driven |
| Fuel_Type | Petrol, Diesel, or CNG |
| Seller_Type | Dealer or Individual |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |
| Selling_Price | Target variable |

## 🤖 Model Performance

The Decision Tree Regressor was optimized using `RandomizedSearchCV`.

| Evaluation Metric | Score |
|---|---:|
| MAE | 0.6774 |
| MSE | 1.1187 |
| RMSE | 1.0577 |
| R² Score | 0.9514 |

**R² Score: 95.14%**

*Note: Error metrics are measured in the dataset's price units (lakhs).*

## 📸 Application Preview

Add your Streamlit application screenshot here:

![Car Price Prediction App](screenshots/streamlit-app.png)

## 📂 Project Structure

```text
CarPricePrediction/
│
├── app.py
├── car_price_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    └── streamlit-app.png
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/VishalShete26/car-price-prediction.git
```

### 2. Navigate to the project directory

```bash
cd car-price-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💾 Model Information

The trained model and fitted categorical encoders are saved together in:

`car_price_model.pkl`

The application loads this file to preprocess user inputs and generate predictions.

## 🎯 Future Improvements

- Experiment with Random Forest and Gradient Boosting.
- Improve model generalization through further evaluation.
- Deploy the application online.
- Add more vehicle information and updated market data.

## 👨‍💻 Author

**Vishal Shete**

- GitHub: [VishalShete26](https://github.com/VishalShete26)

---

*This project was developed for learning and demonstrating Machine Learning model development and deployment using Streamlit.*