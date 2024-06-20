import joblib
import re
import nltk
from nltk.corpus import stopwords

## Load the model and vectorizer
model = joblib.load('D:/Prj/Spam_Detection/spam_model.pkl')
vectorizer = joblib.load('D:/Prj/Spam_Detection/vectorizer.pkl')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    ## Remove non-alphabet characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    ## Convert to lowercase
    text = text.lower()
    ## Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_spam(text):
    text = preprocess_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

if __name__ == "__main__":
    message = input("Enter the message: ")
    prediction = predict_spam(message)
    print(f"Prediction for '{message}': {prediction}")
