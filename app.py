import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import text_processing

model = pickle.load(open('PAmodel.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = {'text':features[0],'title':features[1]}
    prediction = model.predict_one(final_features)

    return render_template('index.html',prediction_text='The News is {}'.format(prediction))
    

if __name__ == "__main__":
    import text_processing    
    app.run(debug=True)
