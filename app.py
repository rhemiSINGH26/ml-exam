from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Load and train the model once at startup
iris = load_iris()
X = iris.data
y = iris.target

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare input and predict
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        # Convert numeric output to nearest class
        class_index = int(round(prediction))
        class_name = iris.target_names[class_index]

        return render_template('index.html', result=f"Predicted Iris species: {class_name}")

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
