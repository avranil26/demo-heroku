#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from sklearn.externals import joblib
import pandas as pd
import numpy as np


# Your API definition
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify("IRIS PREDICTOR")


@app.route('/predict', methods=['POST'])
def predict():
    
    json_ = request.get_json()
    query = pd.get_dummies(pd.DataFrame(json_))
    query = query.reindex(columns=model_columns, fill_value=0)
    prediction = list(lr.predict(query))

    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    
    lr = joblib.load("model.pkl") # Load "model.pkl"
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"    
    app.run(debug=True)

