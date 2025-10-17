# 🧠 Personality Prediction System

A machine learning project built in Python to predict Myers-Briggs Type Indicator (MBTI) personality types based on text data. This model analyzes writing samples to classify an individual's personality into one of the 16 MBTI types.



---

## 🎯 Project Overview

This project explores the fascinating intersection of natural language processing (NLP) and psychology. The primary goal is to build and evaluate a model that can accurately predict a person's personality type from their posts, essays, or other forms of text. This can be useful in fields like personalized marketing, team building, and mental health analysis.

---

## ✨ Features

* **Data Preprocessing:** Includes cleaning text data, removing stop words, and lemmatization.
* **Feature Engineering:** Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
* **Model Training:** Implements a [e.g., Logistic Regression, Naive Bayes, or Random Forest Classifier] model for classification.
* **Prediction API (Optional):** [Mention if you have a simple Flask/FastAPI endpoint to serve predictions.]
* **Evaluation:** Model performance is measured using metrics like Accuracy, Precision, Recall, and F1-Score.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Libraries:**
    * `Pandas`: For data manipulation and analysis.
    * `NLTK`: For natural language processing tasks.
    * `Scikit-learn`: For machine learning models and metrics.
    * `Matplotlib` / `Seaborn`: For data visualization.
    * [Add any other libraries like `Flask`, `Jupyter`, etc.]

---

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/]  https://github.com/Jeeta-sri/Personality-Prediction-System.git
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Make sure you have a `requirements.txt` file in your project by running `pip freeze > requirements.txt`)*

---

## 🚀 How to Use

1.  **Run the training script (if applicable):**
    ```bash
    python train_model.py
    ```

2.  **Make a prediction:**
    Execute the main script to get a personality prediction for a new piece of text.
    ```bash
    python predict.py --text "Your sample text goes here..."
    ```



📊 Dataset

The model was trained on the **[ MBTI (Myers-Briggs) Personality Type Dataset, mbti_1.csv]** from Kaggle. You can find it here[https://www.kaggle.com/datasets/datasnaek/mbti-type]. The dataset contains thousands of text samples, each labeled with one of the 16 personality types.

---

## 📄 License

This project is licensed under the **[e.g., MIT License]**. See the `LICENSE` file for more details.
