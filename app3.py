import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app3 = Flask(__name__)
model3 = pickle.load(open('model3.pkl', 'rb'))

@app3.route('/')
def home():
    return render_template('index3.html')

@app3.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model3.predict(final_features)
    
    output = round(prediction[0], 2)

    return render_template('index3.html', prediction_text='predicted cattle census to be million {}'.format(output))

if __name__ == "__main__":
    app3.run(debug=True)