from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)

p_values=pickle.load(open("pipe_values.pkl","rb"))


@app.route('/')
def home():
    return render_template("heart_failure.html")

@app.route('/predict',methods=["POST"])
def predict():
    age=float(request.form.get("age"))
    gender=request.form.get("gender")
    chest_pain_type=request.form.get("chest_pain_type")
    resting_BP=float(request.form.get("resting_BP"))
    cholesterol=float(request.form.get("cholesterol"))
    blood_sugar=float(request.form.get("blood_sugar"))
    resting_ECG=request.form.get("resting_ECG")
    max_HR=float(request.form.get("max_hr"))
    angina=request.form.get("angina")
    oldpeak=float(request.form.get("oldpeak"))
    st_slope=request.form.get("st_slope")

    input=np.array([age,"gender","chest_pain_type",resting_BP,cholesterol,blood_sugar,"resting_ECG",max_HR,"angina",oldpeak,"st_slope"],dtype=object).reshape(1,11)
    """input = np.array([49,"M","ASY",140,234,0,"Normal",140,"Y",1,"Flat"],dtype=object).reshape(1, 11)"""

    result = p_values.predict(input)

    """return " age :{}, gender : {} ,chest pain type : {}, resting bp : {}, cholesterol : {},blood sugar : {},resting_ECG :{},Max HR:{},Angina:{},oldpeak: {}, st_slope:{}".format(age,gender,chest_pain_type,resting_BP,cholesterol,blood_sugar,resting_ECG,max_HR,angina,oldpeak,st_slope)"""

    if result==1:
        return "chances of heart failure"
    else:
        return "normal"

if __name__ =="__main__":
    app.run(debug=True)