from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__, template_folder="frontend")
CORS(app)

# Load trained model
with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        email_text = data["email"]

        transformed = vectorizer.transform([email_text])

        prediction = model.predict(transformed)[0]

        result = "Spam" if prediction == 1 else "Not Spam"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)