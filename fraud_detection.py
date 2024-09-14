import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
data = pd.read_csv(r'C:\Users\dell\Downloads\project\spam_sms.csv', encoding='ISO-8859-1')

# Check for required columns
if 'label' not in data.columns or 'text' not in data.columns:
    raise ValueError("Dataset must contain 'label' and 'text' columns")

# Prepare features and target
X = data['text']
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
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
print(classification_report(y_test, y_pred, target_names=['Non-fraudulent message', 'Fraudulent message']))

# Save the model
with open(r'C:\Users\dell\Downloads\project\fraud_detect\model\fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model (if needed)
with open(r'C:\Users\dell\Downloads\project\fraud_detect\model\fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)
