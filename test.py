from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and the vectorizer
with open(r'C:\Users\dell\Downloads\project\fraud_detect\model\fraud_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'C:\Users\dell\Downloads\project\fraud_detect\model\tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    message = request.form['message']

    # Convert message to a format your model can use
    data = vectorizer.transform([message])  # Transform the input message

    # Make a prediction
    prediction = model.predict(data)

    # Print the prediction output to check its format
    print("Prediction:", prediction)  # Debugging line

    # Render the result based on prediction
    result = 'Fraudulent' if prediction[0] == 1 else 'Non-fraudulent'
    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
