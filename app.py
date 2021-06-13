import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    d = {'low': 1, 'medium':2,'high':0,'decrease':0,'no change':1,'increase':2,'aggressive':0,'neutral':1,'passive':2,'yes':1,'no':0}
    int_features = [d[x] for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    
    output = round(prediction[0], 4)
    if output ==0:
        output = "No Stress"
    elif output == 1:
        output = "Mild"
    elif output == 2:
        output = "Moderate"
    elif output == 3:
        output = "Severe"
    else:
        output = ""

    return render_template('index.html', prediction_text='Stress Level -- {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)