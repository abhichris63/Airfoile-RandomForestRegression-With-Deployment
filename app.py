import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# request helps to capture the json data which is coming from POSTMAN.
#jsonify converts output into json.


app = Flask(__name__)

#Loading Machine Learning Model from model.pkl file in read byte mode.
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    # Return Home 
    return render_template('home.html') 

# Creation of api
# Api name '/predict_api'
# Method is POST
@app.route('/predict_api',methods=['POST'])
def predict_api():
    # request captures data from POSTMAN and converts into json format
    data = request.json['data']
    # data keys & values pair will be printed
    print(data)
    # Converting dictionary values data to list and passing 
    new_data = [list(data.values())]
    # Passing list of values in 2D Array to predict
    output = model.predict(new_data)[0]
    print(output)
    # Returning the predicted output in (Json)Jsonify Format
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
     # geting form values from HTML
     data = [float(x) for x in request.form.values()]
     # Converting form in 2D array 
     final_feature = [np.array(data)]
     print(data)

    # final_feature is given to predict as input, model is Predicting the output (0 = first output is printed) 
     output = model.predict(final_feature)[0]
    # Printing the output
     print(output)
    # Rendering home.html
     return render_template('home.html', prediction_text = "Airfoil pressure is {}".format(output))


if __name__ == "__main__":
    app.run(debug = True)
