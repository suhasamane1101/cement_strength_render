from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('./model/model.pkl','rb'))
scaler=pickle.load(open('./model/scaler.pkl','rb'))

# creating app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/form')
def index3():
    return render_template('index.html')

@app.route('/dashboard')
def index1():
    return render_template('dashboard.html')


@app.route('/feedback')
def index2():
    return render_template('feedback.html')





@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    #   cement    blastFurnace    flyAsh    water    superplasticizer    courseAggregate    fineaggregate    age
    if request.method=='POST':
        cement = float(request.form['cement'])
        blastFurnace = float(request.form['blastFurnace'])
        flyAsh = float(request.form['flyAsh'])
        water = float(request.form['water'])
        superplasticizer = float(request.form['superplasticizer'])
        courseAggregate = float(request.form['courseAggregate'])
        fineaggregate = float(request.form['fineaggregate'])
        age = int(request.form['age'])

    # transform input features
        features = scaler.transform([[cement, blastFurnace, flyAsh, water, superplasticizer, courseAggregate, fineaggregate, age]])
        prediction = model.predict(features)

        return render_template('index.html', strength=prediction[0])
    else:
        return render_template('index.html')    



# python main
if __name__=="__main__":
    app.run(host="0.0.0.0")
