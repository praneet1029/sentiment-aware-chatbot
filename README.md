# 🤖 Sentiment-Aware Chatbot

## 📌 Project Overview

This project is a **Sentiment-Aware Chatbot** that can understand and respond to the emotional tone of a user's message — detecting if it is **positive**, **negative**, or **neutral**. The chatbot is trained using machine learning techniques and integrated with sentiment analysis to improve user experience in interactive applications.

This is part of my internship assignment and is fully aligned with the course curriculum. It meets all submission guidelines, including proper file organization, model accuracy, and integration of performance metrics.

---

## 🚀 Key Features

- 🎯 Detects sentiment (positive, negative, neutral) in real-time user messages
- 🤝 Responds appropriately based on the sentiment
- 🧠 Trained using **Naive Bayes Classifier** from the NLTK library
- 📊 Evaluated using **accuracy, precision, recall**, and **confusion matrix**
- 💾 Saves trained model and features for reuse
- ☁️ Built and tested using **Google Colab** for easy reproducibility
- ✅ Achieved **accuracy over 80%**, exceeding the minimum requirement of 70%

---

## 📂 Project Structure

| File/Folder         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `sentiment_model.ipynb` | Jupyter Notebook containing the full pipeline: data prep, training, testing, and chatbot simulation |
| `saved_model.pkl`        | Saved sentiment classifier model using `joblib`                         |
| `word_features.pkl`      | Saved feature extractor used for text classification                   |
| `requirements.txt`       | List of Python libraries required to run the project                   |
| `README.md`              | Project documentation file (this file)                                 |

---

## 🛠️ Tools & Libraries Used

- **Python 3**
- **Google Colab** (for training & testing)
- `nltk` – for natural language processing & Naive Bayes Classifier
- `scikit-learn` – for model evaluation (metrics like precision, recall, confusion matrix)
- `joblib` – for saving/loading model files
- `matplotlib`, `seaborn` – for plotting evaluation metrics

---

## 🎓 Learning Objectives

- Understand how sentiment analysis can be integrated into chatbot systems
- Apply text preprocessing and feature extraction techniques
- Train and evaluate a machine learning model for NLP classification
- Improve chatbot response relevance using sentiment context
- Organize a complete project for professional submission

---

## 📈 Performance Evaluation

The model was trained on a labeled dataset of sentiment-tagged phrases and evaluated using multiple metrics:

| Metric           | Value    |
|------------------|----------|
| Accuracy         | 80%+     |
| Precision        | High     |
| Recall           | High     |
| Confusion Matrix | Included in Notebook |

> Evaluation graphs and confusion matrix plots are available in the notebook (`sentiment_model.ipynb`).

---

## 💬 How the Chatbot Works

1. **Input**: User types a message.
2. **Processing**: 
   - The message is cleaned and preprocessed (removing stopwords, lowercasing, etc.).
   - Features are extracted from the message text.
   - The trained model predicts the sentiment.
3. **Response**:
   - If sentiment is **positive** → friendly/happy message
   - If sentiment is **negative** → comforting/supportive message
   - If sentiment is **neutral** → basic/default response

---

