# fraud_detect
# Fraudulent Message Detection App

## Overview
The Fraudulent Message Detection Application is designed to detect and classify SMS messages as either legitimate or fraudulent. This web-based application uses a machine learning model to perform real-time analysis of incoming messages and classify them accordingly.

## Features
- *SMS Fraud Detection*: The application identifies whether a given message is fraudulent or safe.
- *Machine Learning Model*: The app uses a trained model with 96.23% accuracy to classify messages.
- *Real-time Prediction*: The app allows users to input text and get instant feedback on the nature of the message.
- *Web-based Interface*: Built using Flask, it provides an easy-to-use web interface for users.

## Tech Stack
- *Frontend*: HTML, CSS, JavaScript (for the user interface)
- *Backend*: Flask (Python-based web framework)
- *Machine Learning*: scikit-learn, pandas, and other libraries for building and evaluating the model.
- *Database*: SQLite or any other database (optional, for storing message logs).

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- pip (Python package manager)

### Installation

1. *Clone the Repository*
    bash
    git clone https://github.com/Basveshwari835/fraudulent-message-detection-app.git
    cd fraudulent-message-detection-app
    

2. *Create a Virtual Environment*
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    

3. *Install Dependencies*
    bash
    pip install -r requirements.txt
    

4. *Download or Create a Dataset*
   You can use your own dataset for fraudulent message detection or download a preprocessed dataset (e.g., from [Kaggle](https://www.kaggle.com)).

5. *Train the Model*
    Before running the application, train your model using the dataset. Ensure the model is saved to be used later in the app.
    bash
    python train_model.py
    

6. *Run the Application*
    After successfully training the model, run the Flask app.
    bash
    python test.py
    

7. *Access the App*
    Open your web browser and go to http://127.0.0.1:5000 to access the application.

## Project Structure
## Model Information
The model used for detecting fraudulent messages is based on Natural Language Processing (NLP) techniques. It is trained using a dataset of labeled SMS messages and uses methods such as:
- *TF-IDF Vectorization*: To convert text into numerical features.
- *Logistic Regression / SVM / Random Forest*: For message classification.

## Future Improvements
- Implement a better UI/UX design for the web interface.
- Add multi-language support for message detection.
- Deploy the application on a cloud platform such as Heroku or AWS.

## Contributors
- [Basaveshwari Malipatil ](https://github.com/Basaveshwari835)
- [Sanjana s Kadakbhavi](https://github.com/Sanjanask13)
- [Annapurna b](https://github.com/Anhiii)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
