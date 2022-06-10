import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os


 #Initialize the flask App
# Create flask app
flask_app = Flask(__name__)
model1 = pickle.load(open("model_pickle.pkl", "rb"))

model2 = pickle.load
(open('model_picklenn.pkl', 'rb'))


# ''' @app.route('/')
# def main(): 
   
#     return render_template('homepage.html')


# @app.route('/predict')
# def result(): '''
    
@flask_app.route("/")
def Home():
    return render_template("homepage.html")

@flask_app.route("/predictsvm", methods = ["POST"])
def predictsvm():
 
    with open('resltsvm.txt', 'r') as f: 
	        return render_template('resultsvm.html', text=f.read())
   


@flask_app.route("/predictnn", methods = ["POST"])

def predictnn():
    with open('resltnn.txt', 'r') as f: 
	        return render_template('resultnn.html', text=f.read())

if __name__ == "__main__":
    flask_app.run(debug=True)
