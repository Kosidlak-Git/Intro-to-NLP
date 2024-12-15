import warnings
import re
import nltk
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest
import os

nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

if not os.path.exists("static"):
    os.mkdir("static")

stop_words = set(stopwords.words("english"))
punct = punctuation + "\n" + "—" + "“" + "," + "”" + "‘" + "-" + "’"

warnings.filterwarnings("ignore")


def preprocess_and_summarize(text):
    # Preprocessing
    text = text.lower()
    text = re.sub(r"([.?!])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove punctuation
    text_punctuation = "".join([char for char in text if char not in punct])
    sentences = sent_tokenize(text)

    # Remove stopwords
    cleaned_text = " ".join(
        word for word in word_tokenize(text_punctuation) if word not in stop_words
    )

    # Get word frequencies
    word_frequency = {}
    for word in word_tokenize(cleaned_text):
        if word not in stop_words:
            if word in word_frequency:
                word_frequency[word] += 1
            else:
                word_frequency[word] = 1

    # Normalize frequencies
    if word_frequency:
        max_frequency = max(word_frequency.values())
        for word in word_frequency:
            word_frequency[word] = word_frequency[word] / max_frequency

    # Score sentences
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent):
            if word in word_frequency:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequency[word]
                else:
                    sentence_scores[sent] += word_frequency[word]

    # Create summary
    summary_length = int(len(sentences) * 0.2)
    summary = " ".join(
        nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    )

    original_words = [word for word in word_tokenize(text) if word.isalnum()]
    summary_words = [word for word in word_tokenize(summary) if word.isalnum()]
    original_word_count = len(original_words)
    summary_word_count = len(summary_words)
    tokenized_sentence_count = len(sentences)

    return (
        original_word_count,
        summary_word_count,
        summary,
        tokenized_sentence_count,
        word_frequency,
    )


def create_visual(word_frequency):
    # Get top 100 words
    top_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:100]
    words, frequencies = zip(*top_words)

    plt.figure(figsize=(25, 10))
    plt.bar(words, frequencies)
    plt.title("Top 100 Words by Frequency")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = "static/top_words.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            (
                original_count,
                summary_count,
                summary,
                tokenized_sentence_count,
                word_frequency,
            ) = preprocess_and_summarize(text)
            plot_path = create_visual(word_frequency)

            return render_template(
                "index.html",
                original_count=original_count,
                summary_count=summary_count,
                summary=summary,
                tokenized_sentence_count=tokenized_sentence_count,
                plot_path=plot_path,
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
