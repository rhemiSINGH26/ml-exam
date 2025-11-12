from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("random_forest_model.pkl")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    feature = [78, 113, 22.4]
    input = np.array(feature).reshape(1, -1)
    prediction = model.predict(input)[0]
    if prediction == 0:
        return jsonify({'prediction': 'non-diabetic'})
    else:
        return jsonify({'prediction': 'diabetic'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
