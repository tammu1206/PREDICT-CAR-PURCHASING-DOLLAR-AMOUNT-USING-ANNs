# Car Purchase Amount Prediction using Artificial Neural Networks (ANNs)

## Project Overview
This project involves building an Artificial Neural Network (ANN) model to predict the amount of money a customer is willing to pay for a car based on their demographic and financial information. This is a regression task, as the goal is to predict a continuous numerical value — the car purchase amount.

## Problem Statement
You are a car salesman, and you would like to predict the total dollar amount that customers are willing to pay based on the following attributes:
- **Customer Name**
- **Customer E-mail**
- **Country**
- **Gender**
- **Age**
- **Annual Salary**
- **Credit Card Debt**
- **Net Worth**

The model will output a prediction for the **Car Purchase Amount**.

---

## Dataset
- The dataset contains customer demographic and financial information.
- Some features like **Customer Name** and **Customer E-mail** will be dropped since they are irrelevant for prediction.
- The target variable is the **Car Purchase Amount**.

### Data Preprocessing
- Handling missing or incorrect data.
- Encoding categorical variables (e.g., Gender, Country).
- Feature scaling using standardization.

---

## Model Architecture
- The model is built using an Artificial Neural Network (ANN).
- It consists of the following layers:
  - Input Layer
  - Hidden Layers (using ReLU activation)
  - Output Layer (using linear activation for regression)
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Evaluation Metric: Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE)

---

## Requirements
To run this project, ensure you have the following libraries installed:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow or Keras
- Scikit-Learn

Install them using the following command:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-repo/car-purchase-prediction.git
cd car-purchase-prediction
```
2. Run the Python script:
```bash
python car_purchase_prediction.py
```
3. The model will train on the dataset and display evaluation metrics.
4. You can adjust hyperparameters in the script to optimize performance.

---

## Results
- After training, the model will provide predictions for car purchase amounts.
- Visualizations of loss curves and predicted vs actual values will be generated.

---

## Future Improvements
- Implement hyperparameter tuning for better accuracy.
- Use additional feature engineering techniques.
- Deploy the model using a web or mobile app for real-world use.

---

## Conclusion
This project demonstrates the application of Artificial Neural Networks for regression tasks. By predicting car purchase amounts, it provides valuable insights to car sales businesses, assisting them in understanding customer purchasing behavior.

---
