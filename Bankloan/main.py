from flask import Flask, render_template,request
import numpy as np
import pickle
app=Flask(__name__)

model=pickle.load(open('rf_model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    gender=request.form['gender']
    married=request.form['married']
    dependents=request.form['dependents']
    education=request.form['education']
    self_employed=request.form['self_employed']
    applicant_income=float(request.form['applicant_income'])
    coapplicant_income=float(request.form['coapplicant_income'])
    loan_amount=float(request.form['loan_amount'])
    credit_history=int(request.form['credit_history'])
    property_area=request.form['property_area']

    gender = {'MALE': 1, 'FEMALE': 0} [gender]
    married = {'YES': 1, 'NO': 0} [married]
    dependents = {'0': 0, '1': 1, '2':2, '3+': 3} [dependents]
    education = {'GRADUATES': 0, 'NOT GRADUATES': 1}[education]
    self_employed = {'YES': 1, 'NO': 0}[self_employed]
    property_area = {'SEMIURBAN': 0, 'URBAN': 1, 'RURAL': 2} [property_area]

    scaled_data=scaler.transform([[applicant_income,coapplicant_income,loan_amount]])

    applicant_income=scaled_data[0][0]
    coapplicant_income = scaled_data[0][1]
    loan_amount = scaled_data[0][2]

    features= np.array([[gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,credit_history,property_area]])

    prediction =model.predict(features)

    if prediction[0]==1:
        result = 'LOAN APPROVED'
    else:
        result = 'NOT LOAN APPROVED'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
