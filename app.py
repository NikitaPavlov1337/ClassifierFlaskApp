import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

#Load Model
model = pickle.load(open("out/model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    features = [int(x) for x in request.form.values()]
    features = [np.array(features)]
    prediction = model.predict(features)
    if prediction == True:
        return render_template("index.html", prediction_text=f"Классификация клиента: надежный")
    else:
        return render_template("index.html", prediction_text = f"Классификация клиента: убыточный для кампании ")

if __name__ == "__main__":
    flask_app.run(debug=True)
