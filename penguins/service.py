import pickle
import numpy as np
from flask import Flask, jsonify, request
import pandas as pd

app = Flask('penguin-predict')

def predict_single(penguin, vectorizer, scaler, model):
    penguin_categoricas = vectorizer.transform([{
        'island': penguin['island'],
        'sex': penguin['sex']
    }]) 

    penguin_numericas_df = pd.DataFrame([[
        penguin['bill_length_mm'],
        penguin['bill_depth_mm'],
        penguin['flipper_length_mm'],
        penguin['body_mass_g']
    ]], columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])    

    penguin_numericas = scaler.transform(penguin_numericas_df)
    penguin_completo = np.hstack([penguin_categoricas, penguin_numericas])  
    y_pred = model.predict(penguin_completo)[0]
    y_prob = model.predict_proba(penguin_completo)[0][y_pred]
    return (y_pred, y_prob)


@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('models/lr.pck', 'rb') as f:
        dv,sc, model = pickle.load(f)
    penguin = request.get_json()
    print(penguin)
    penguin, prediction = predict_single(penguin, dv, sc, model)

    result = {
        'penguin': bool(penguin),
        'penguin_probability': float(prediction)
    }

    return jsonify(result)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('models/svm.pck', 'rb') as f:
        dv,sc, model = pickle.load(f)
    penguin = request.get_json()
    print(penguin)
    penguin, prediction = predict_single(penguin, dv, sc, model)

    result = {
        'penguin': bool(penguin),
        'penguin_probability': float(prediction)
    }

    return jsonify(result)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('models/dt.pck', 'rb') as f:
        dv,sc, model = pickle.load(f)
    penguin = request.get_json()
    print(penguin)
    penguin, prediction = predict_single(penguin, dv, sc, model)

    result = {
        'penguin': bool(penguin),
        'penguin_probability': float(prediction)
    }

    return jsonify(result)


@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('models/knn.pck', 'rb') as f:
        dv,sc, model = pickle.load(f)
    penguin = request.get_json()
    print(penguin)
    penguin, prediction = predict_single(penguin, dv, sc, model)

    result = {
        'penguin': bool(penguin),
        'penguin_probability': float(prediction)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8000)  


"""
curl --request POST "http://127.0.0.1:8000/predict" \
--header "Content-Type: application/json" \
--data-raw "{\
    \"island\": \"Torgersen\",\
    \"sex\": \"FEMALE\",\
    \"bill_length_mm\": 39.1,\
    \"bill_depth_mm\": 18.7,\
    \"flipper_length_mm\": 181.0,\
    \"body_mass_g\": 3800.0\
}"

{
  "penguin": false,
  "penguin_probability": 0.0025385558731211982
}
"""