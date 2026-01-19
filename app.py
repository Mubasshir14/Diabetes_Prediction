import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)


def predict_diabetes(
    Pregnancies, Glucose, BloodPressure, SkinThickness,
    Insulin, BMI, DiabetesPedigreeFunction, Age
):

    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]],
    columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])


    prediction = model.predict(input_df)[0]

    if prediction == 1:
        return "⚠️ Diabetes Detected"
    else:
        return "✅ No Diabetes Detected"


inputs = [
    gr.Slider(0, 20, step=1, value=2, label="Pregnancies"),
    gr.Slider(50, 200, step=1, value=120, label="Glucose Level"),
    gr.Slider(40, 130, step=1, value=70, label="Blood Pressure"),
    gr.Slider(0, 100, step=1, value=20, label="Skin Thickness"),
    gr.Slider(0, 900, step=10, value=80, label="Insulin"),
    gr.Slider(10, 70, step=0.1, value=25.0, label="BMI"),
    gr.Slider(0.0, 3.0, step=0.01, value=0.5, label="Diabetes Pedigree Function"),
    gr.Slider(10, 100, step=1, value=30, label="Age")
]


app = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="🩺 Diabetes Prediction App",
    description=(
        "Enter patient medical information to predict the likelihood of diabetes.\n\n"
        "This app uses a Machine Learning model trained on the Pima Indians Diabetes Dataset."
    ),
    theme="soft"
)


app.launch(share=True)
