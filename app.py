from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    prediction = None
    message = ""
    confidence = None
    error = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            error = "Please enter a message to analyze."
        else:
            message_tfidf = vectorizer.transform([message])

            prediction = model.predict(message_tfidf)[0]

            probabilities = model.predict_proba(message_tfidf)[0]
            confidence = round(max(probabilities) * 100, 2)

    return render_template(
        "analyze.html",
        prediction=prediction,
        message=message,
        confidence=confidence,
        error=error
    )


@app.route("/analytics")
def analytics():

    metrics = {
        "accuracy": 97,
        "spam_precision": 98,
        "spam_recall": 83,
        "spam_f1": 90
    }

    confusion_matrix = {
        "true_ham": 963,
        "false_spam": 3,
        "false_negative": 25,
        "true_spam": 124
    }

    return render_template(
        "analytics.html",
        metrics=metrics,
        confusion_matrix=confusion_matrix
    )


@app.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")


if __name__ == "__main__":
    app.run(debug=True)