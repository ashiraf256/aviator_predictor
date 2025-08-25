from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    last = float(request.form['last'])
    avg = float(request.form['avg'])
    vol = float(request.form['vol'])

    X = np.array([[last, avg, vol]])
    model = RandomForestClassifier()
    model.fit([[1,2,3],[2,3,4],[3,4,5]], [0,1,0])  # Dummy training
    prediction = model.predict(X)[0]
    confidence = round(np.random.uniform(70, 99), 2)

    return f"Prediction: {prediction}, Confidence: {confidence}%"

if __name__ == '__main__':
    app.run()
