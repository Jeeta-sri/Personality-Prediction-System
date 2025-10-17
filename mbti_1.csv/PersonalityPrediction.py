# Step 1: Import several libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load the dataset - use existing MBTI dataset (you can download it from online sources)
data = pd.read_csv('mbti_1.csv')  # This file needs to be in your notebook directory

# Step 3: Prepare data inputs and outputs
X = data['posts']  # Text data
y = data['type']   # MBTI personality labels

# Step 4: Encode labels to numeric
le = LabelEncoder()
y = le.fit_transform(y)

# Step 5: Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Convert text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Step 8: Predict on test data
y_pred = model.predict(X_test_tfidf)

# Step 9: Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Step 10: Function to predict personality type of new text input
def predict_personality(text):
    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)
    pred_label = le.inverse_transform(pred)[0]
    return pred_label

# Example Prediction
new_text = "Stay strong and never ever gove up on your dreams!"
pred_type = predict_personality(new_text)
print(f'Predicted MBTI personality type for the input text: {pred_type}')
