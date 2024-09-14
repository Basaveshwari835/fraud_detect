import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset, replace the path with your actual dataset path
data = pd.read_csv(r'C:\Users\dell\Downloads\project\spam_sms.csv',encoding='ISO-8859-1')

# Ensure 'label' and 'text' columns exist
if 'label' not in data.columns or 'text' not in data.columns:
    raise ValueError("Dataset must contain 'label' and 'text' columns")

# Separate features and target variable
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

from sklearn.metrics import classification_report

# Add this to your existing code
print(classification_report(y_test, y_pred, target_names=['Non-fraudulent', 'Fraudulent']))

""" # After predicting y_pred
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
new_threshold = 0.3  # Adjust this threshold based on your needs
y_pred_new_threshold = (y_prob >= new_threshold).astype(int)

# Recalculate and print classification report with new threshold
print(classification_report(y_test, y_pred_new_threshold, target_names=['Non-fraudulent', 'Fraudulent']))

# Mapping 0 to 'ham' and 1 to 'spam' to match the labels in y_test
y_pred_labels = ['ham' if pred == 0 else 'spam' for pred in y_pred_new_threshold]

# Now, run the classification report with the corrected labels
print(classification_report(y_test, y_pred_labels, target_names=['ham', 'spam'])) """
# After you calculate y_prob and y_pred_new_threshold
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
new_threshold = 0.3  # You set the threshold here
y_pred_new_threshold = (y_prob >= new_threshold).astype(int)

# Convert the numeric predictions back to 'ham' and 'spam' to match y_test
y_pred_labels = ['ham' if pred == 0 else 'spam' for pred in y_pred_new_threshold]

# Now, print the classification report with the correct label format
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_labels, target_names=['ham', 'spam']))

from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(_name_)

# Load your trained model
model = pickle.load(open('model/fraud_detection_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    message = request.form['message']

    # Convert message to a format your model can use
    data = pd.DataFrame([message], columns=['message'])

    # Make a prediction
    prediction = model.predict(data['message'])

    # Render the result
    if prediction[0] == 1:
        return render_template('result.html', result='Fraudulent')
    else:
        return render_template('result.html', result='Non-fraudulent')

if _name_ == "_main_":
    app.run(debug=True)
