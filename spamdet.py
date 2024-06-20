import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re
import joblib

## Load dataset
data_path = 'D:/Prj/Spam_Detection/data/spam.csv'
data = pd.read_csv(data_path, encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

## Explore dataset
print(data.head())

## Preprocess text
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

data['text'] = data['text'].apply(preprocess_text)

## Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

## Vectorize text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

## Print vocabulary size
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

## Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

## Make predictions
y_pred = model.predict(X_test_vec)

## Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

## Save model and vectorizer
joblib.dump(model, 'D:/Prj/Spam_Detection/spam_model.pkl')
joblib.dump(vectorizer, 'D:/Prj/Spam_Detection/vectorizer.pkl')

print("Model and vectorizer saved.")