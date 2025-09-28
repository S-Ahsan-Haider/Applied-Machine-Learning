from flask import Flask, request
import numpy as np
import pickle
import sklearn

app = Flask(__name__)

model = open("admission.pkl","rb")
reg = pickle.load(model)

scaler = open("scaler.pkl","rb")
sca = pickle.load(scaler)

@app.route("/")
def hello_world():
    return "<p>Predictor is Online!</p>"

@app.route("/predict", methods = ['POST'])
def prediction():
    student = request.get_json()
    print(student)

    if student['research'] == 'Yes':
        research = 1
    elif student['research'] == 'No':
        research = 0

    gre_score = student['gre_score']
    toefl_score = student['toefl_score']
    univ_rating = student['univ_rating']
    sop = student['sop']
    lor = student['lor']
    cgpa = student['cgpa']

    features = np.array([[gre_score, toefl_score, univ_rating, sop, lor, cgpa, research]])
    
    sca_feat = sca.transform(features)
    
    result = reg.predict(sca_feat)
    
    pred = f"{result[0] * 100:.2f}%"

    return{"Your Chance To Get Admission is": pred}