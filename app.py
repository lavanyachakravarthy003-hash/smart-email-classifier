from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
app = Flask(__name__)
CORS(app)

model = pickle.load(open('model/spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Smart Email Classifier Running!"

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json['email']
    
    transformed = vectorizer.transform([email_text]).toarray()
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"prediction": result})


  

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)