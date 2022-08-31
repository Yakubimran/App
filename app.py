import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os


#vectorizer
vectorizer_model=open(os.path.join("static/vectorizer.pkl"),'rb')
vectorizer=joblib.load(vectorizer_model)

app = Flask(__name__)

def get_keys(val,my_dict):
    for key, value in my_dict.items():
        if val ==value:
            return key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method== "POST":
            rawtext = request.form ['rawtext']
            vectorizer_text = vectorizer.transform([rawtext]).toarray()
            model = open(os.path.join("static/finalized_model.pkl"),"rb")
            news=joblib.load(model)
#prediction
    prediction_labels ={"Verified":1,"Unverified":0}
    prediction= news.predict(vectorizer_text)
    final_result =get_keys(prediction,prediction_labels)

    return render_template("index.html",rawtext=rawtext.upper(),final_result=final_result)

if __name__ == "__main__":
    app.run(debug=True)
