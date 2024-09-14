import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load the dataset
data = pd.read_csv(r'C:\Users\dell\Downloads\project\fraud_detect\spam_sms.csv', encoding='ISO-8859-1')

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
print(classification_report(y_test, y_pred, target_names=['Non-fraudulent', 'Fraudulent']))

# Save the trained model and vectorizer
with open(r'C:\Users\dell\Downloads\project\fraud_detect\model\fraud_detection_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open(r'C:\Users\dell\Downloads\project\fraud_detect\model\tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
