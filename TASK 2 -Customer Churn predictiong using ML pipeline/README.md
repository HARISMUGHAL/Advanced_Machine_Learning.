
# 📊 Telco Customer Churn Prediction

This project uses **machine learning** and **Streamlit** to predict customer churn for a telecommunications company. It includes a complete pipeline for preprocessing, model training with hyperparameter tuning, and a user-friendly web interface.

---

## 🛠️ Technologies Used

- Python
- pandas
- scikit-learn
- joblib
- Streamlit

---

## 📂 Project Structure

```
├── telco-churn-dataset.csv       # Dataset file
├── train_model.py                # ML model training and saving script
├── app.py                        # Streamlit app for prediction
├── telco_churn_pipeline.pkl      # Trained and saved ML pipeline
└── README.md                     # Project documentation
```

---

## 📌 Features

- Full machine learning pipeline using `Pipeline` and `ColumnTransformer`
- Scales numeric features and one-hot encodes categorical features
- Uses `RandomForestClassifier` with `GridSearchCV` for hyperparameter tuning
- Model is saved using `joblib`
- Interactive UI built using Streamlit for live prediction

---

## 📊 Dataset Info

The dataset contains information about telecom customers, including:

- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, etc.
- **Billing:** `MonthlyCharges`, `TotalCharges`, `Contract`, `PaymentMethod`
- **Target Variable:** `Churn` (Yes/No)

---

## 🧠 Model Training

```bash
python train_model.py
```

- Drops missing values and converts `TotalCharges` to numeric
- Splits the data into training and testing sets
- Trains a `RandomForestClassifier` with cross-validation
- Saves the best model as `telco_churn_pipeline.pkl`

---

## 🔮 Streamlit App Usage

To run the Streamlit app:

```bash
streamlit run app.py
```

- Choose input values from dropdowns and number fields
- Click the **Predict Churn** button
- See result: `Yes` or `No` for churn prediction

---

## 💾 Requirements

Install required packages:

```bash
pip install pandas scikit-learn streamlit joblib
```

---

## 📈 Sample Output

```
Best Accuracy: 0.82
Best Parameters: {
  'classifier__max_depth': 20,
  'classifier__min_samples_split': 2,
  'classifier__n_estimators': 200
}
Model saved as 'telco_churn_pipeline.pkl'
```

---

## 👤 Author

**Haris Mughal**  
ML & Streamlit Developer

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify.
