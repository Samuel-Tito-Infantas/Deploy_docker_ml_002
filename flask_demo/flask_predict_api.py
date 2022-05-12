#!/usr/bin/env teste_001_env
# -*- coding: utf-8 -*-

"""
Created on 26/04/2022

@author: sti

"""
## 1) Import lib

import pandas as pd
import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np




## 2) Load model 

with open("./rf.pkl",'rb') as model_file:
    model = pickle.load(model_file)

## 3) Star Flask app

app =  Flask(__name__)
swagger = Swagger(app)

@app.route('/predict')
def predict_iris():
    """ This endpoint returning a predict of iris.
    ---
    parameters:
      - name: s_length
        in: query
        type: number
        required: true
      - name: s_width
        in: query
        type: number
        required: true
      - name: p_length
        in: query
        type: number
        required: true
      - name: p_width
        in: query
        type: number
        required: true
    responses:
      200:
        description: "text"
        content:
          application/json:
          schema:
            type: object    
    """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")

    prediction = model.predict(np.array([[s_length ,s_width
                                         ,p_length ,p_width ]]))
    
    return str(prediction)


@app.route('/predict_file', methods = ["POST"])
def predict_iris_file():
    """ This endpoint returning a predict of iris with file
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: "text"
        content:
          application/json:
          schema:
            type: object
    """
    input_data = pd.read_csv(request.files.get("input_file"),header=None)
    prediction = model.predict(input_data)
    
    return str(list(prediction))


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
