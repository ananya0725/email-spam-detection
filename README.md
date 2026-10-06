# 📧 Email Spam Detection Using Machine Learning

## 📌 Project Overview

Email Spam Detection is a machine learning-based web application that automatically classifies email messages as **Spam** or **Legitimate (Ham)**.

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert email text into numerical features and **Complement Naive Bayes** to classify the message.

The project provides a web interface built using **Flask, HTML, CSS, and Python**.

---

## 🎯 Objectives

* Automatically detect spam emails.
* Classify messages as Spam or Legitimate.
* Convert text into numerical features using TF-IDF.
* Train a machine learning classification model.
* Provide a simple web interface for testing emails.
* Display model performance and evaluation metrics.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* TF-IDF Vectorization
* Complement Naive Bayes

### Data Processing

* Pandas
* NumPy

### Web Development

* Flask
* HTML
* CSS

### Development Tools

* Visual Studio Code
* Git & GitHub

---

## ⚙️ How the System Works

The system follows these steps:

```text
Email Message
     ↓
Text Processing
     ↓
TF-IDF Vectorization
     ↓
Complement Naive Bayes
     ↓
Spam / Legitimate
```

### 1. Email Input

The user enters an email message through the web application.

### 2. TF-IDF Vectorization

The email text is converted into numerical features using TF-IDF.

### 3. Classification

The trained Complement Naive Bayes model analyzes the features.

### 4. Prediction

The system classifies the message as either:

* 🚨 Spam
* ✅ Legitimate

---

## 📊 Model Performance

The model was evaluated using a held-out test set.

| Metric         | Score |
| -------------- | ----: |
| Accuracy       |   97% |
| Spam Precision |   98% |
| Spam Recall    |   83% |
| Spam F1-Score  |   90% |

### Confusion Matrix

|             | Predicted Ham | Predicted Spam |
| ----------- | ------------: | -------------: |
| Actual Ham  |           963 |              3 |
| Actual Spam |            25 |            124 |

The model was trained and evaluated using the SMS Spam Collection dataset included in this project.

---

## 📂 Project Structure

```text
Email spam/
│
├── app.py
├── train_model.py
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│   ├── index.html
│   ├── analyze.html
│   ├── analytics.html
│   └── how-it-works.html
│
└── static/
    └── css/
        └── style.css
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd "Email spam"
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

---

## 🧪 Testing

The application can be tested using:

* Spam messages
* Legitimate messages
* Empty input
* Different email lengths and formats

The Analytics page displays the model's evaluation metrics and confusion matrix.

---

## 🔮 Future Improvements

* Improve detection of more complex spam messages.
* Add support for larger and more diverse datasets.
* Add email file upload functionality.
* Improve text preprocessing.
* Add user authentication.
* Deploy the application online.
* Add more machine learning models for comparison.

---

## 👩‍💻 Project

**Email Spam Detection Using Machine Learning**

Built using Python, Flask, TF-IDF, and Complement Naive Bayes.
