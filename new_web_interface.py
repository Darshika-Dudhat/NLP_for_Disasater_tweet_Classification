from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Logistic Regression model
model_lr = joblib.load('./logistic_regression_model.pkl')

# Load the TF-IDF vectorizer (make sure to use the same vectorizer used during training)
tfidf_vectorizer = joblib.load('tfidf_vectorizer_lr.pkl')  # Make sure the vectorizer is saved previously

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the tweet text from the form
        tweet_text = request.form['tweet_text']

        # Vectorize the tweet text using the same TF-IDF vectorizer
        tweet_vectorized = tfidf_vectorizer.transform([tweet_text])

        # Make a prediction using the Logistic Regression model
        prediction = model_lr.predict(tweet_vectorized)
        
        # Map prediction to human-readable labels
        result_label = 'Disaster' if prediction[0] == 1 else 'Non-Disaster'

        # Return the result as a JSON response
        result = {'prediction': result_label}
        # return jsonify(result)

        # Return the result as a response and render the page with the prediction
        return render_template('index.html', prediction=result_label)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
