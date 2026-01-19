# 🩺 Diabetes Prediction System – Project Overview

This project implements an end-to-end machine learning pipeline to predict the likelihood of diabetes using medical and demographic features. The system is designed following standard data science and MLOps practices, from data preprocessing and model training to deployment through a web interface.

---

## 📊 Dataset

The model is trained on the **Pima Indians Diabetes Dataset**, which contains medical attributes such as pregnancies, glucose level, blood pressure, insulin, BMI, diabetes pedigree function, and age. The target variable indicates whether a patient has diabetes or not.

---

## 🧹 Data Preprocessing

Several preprocessing steps were applied to ensure data quality and model performance:

- Invalid zero values in medical features were replaced with missing values.
- Missing values were handled using **median imputation**.
- All numerical features were standardized using **feature scaling**.
- Features and target labels were separated, followed by a **stratified train–test split**.

---

## 🧠 Model Training

A **Logistic Regression** classifier was selected due to its effectiveness in binary classification problems and interpretability in healthcare applications.

The model was trained using a **Scikit-learn Pipeline**, which integrates preprocessing and model training into a single workflow.

---

## 🔁 Model Validation

To ensure robustness, **5-fold cross-validation** was applied on the training data. Model performance was evaluated using **accuracy scores**.

---

## ⚙️ Hyperparameter Tuning

**GridSearchCV** was used to optimize model hyperparameters, allowing the system to automatically select the best performing configuration based on cross-validated accuracy.

---

## 🏆 Best Model Selection

The best model obtained from hyperparameter tuning was selected and evaluated on unseen test data using the following metrics:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

---

## 💾 Model Saving

The final trained pipeline (including preprocessing steps and the classifier) was saved as a `.pkl` file. This allows the model to be reused for predictions without retraining.

---

## 🌐 Web Application

A user-friendly **Gradio web interface** was developed to deploy the trained model. Users can input patient medical data through interactive sliders and receive instant diabetes prediction results.

---

## 🚀 Conclusion

This project demonstrates a complete machine learning workflow, from raw data handling and model optimization to real-time deployment, making it suitable for academic, demonstration, and beginner-level production use.
