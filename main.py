from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

columns = ['pop','city','lat','capacity','container','price','brand']

with open("model.pkl","rb") as f:
    model_body = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['GET','POST'])
def submit_form():
    pop = float(request.form['pop'])
    city = float(request.form['city'])
    lat = float(request.form['lat'])
    capacity = float(request.form['capacity'])
    container = float(request.form['container'])
    price = float(request.form['price'])
    brand = float(request.form['brand'])
    new_data = [pop,city,lat,capacity,container,price,brand]
    #new_data=pd.DataFrame(new_data,columns=columns)
    #new_input=new_data.values
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    new_input=np.array([new_data])
    c= scaler.transform(new_input)
    reshaped_input = c.reshape(1, -1)
    res = model_body.predict(reshaped_input)
    response = {'code':200,'status':'OK',
                'result':str(res[0])}
    return jsonify(response)

if __name__== '__main__':
    app.run(debug=True)