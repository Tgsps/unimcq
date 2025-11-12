"""
MCQ Quiz Generator Web Application
---------------------------------

This Flask-based web application allows teachers or students to upload a PDF
containing lecture notes or textbook chapters and automatically generate a
multiple‑choice quiz from the extracted text.  The app uses simple natural
language processing (NLP) techniques to pull out nouns from the text and
construct fill‑in‑the‑blank questions.  Students can then answer the
questions, and the app will grade their responses and provide feedback on the
correct answers.

Key features:

* Upload a PDF file and specify how many questions to generate.
* Extract text from the PDF using PyPDF2 (a pure‑Python PDF parser).
* Identify candidate nouns with NLTK and create multiple‑choice questions by
  hiding the noun in its sentence and supplying distractor options.
* Present a quiz form for students to answer and then grade the responses.
* Display a results page that highlights which questions were answered
  correctly and reveals the correct answer for those that were missed.

Because we are running in an isolated environment without access to large
language models, this implementation relies on heuristic rules rather than
advanced AI.  Nevertheless, it demonstrates a complete end‑to‑end workflow
for automated quiz generation from PDF content.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import List

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize, pos_tag
import PyPDF2


app = Flask(__name__)
app.secret_key = "supersecretkey"  # needed for flashing messages


@dataclass
class MCQ:
    """Simple data structure to hold the components of a multiple choice question."""

    id: int
    question: str
    options: List[str]
    answer: str


def ensure_nltk_data() -> None:
    """Ensure that the necessary NLTK corpora are downloaded.

    The NLTK library ships without its data files by default.  If the
    required corpora (tokenizers, taggers, stopwords) are not present,
    this function downloads them on demand.  The download calls are
    idempotent: if the data already exists, nothing happens.
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger")
    except LookupError:
        nltk.download("averaged_perceptron_tagger")

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    """Read all textual content from a PDF file.

    Args:
        file_stream: A file‑like object containing PDF data.

    Returns:
        A string with all extracted text concatenated across pages.
    """

    reader = PyPDF2.PdfReader(file_stream)
    text_chunks: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            # If extraction fails for a page, skip it gracefully.
            page_text = ""
        text_chunks.append(page_text)
    return "\n".join(text_chunks)


def generate_mcq_questions(text: str, num_questions: int = 5) -> List[MCQ]:
    """Create a list of multiple‑choice questions from provided text.

    The algorithm performs the following steps:
    1. Tokenise the text into sentences.
    2. Identify all nouns in the entire text and compile a pool of candidate
       answer words (excluding common stopwords and duplicates).
    3. Iterate through sentences and select the first noun that appears as
       the answer for that question.  Replace the noun in the sentence with
       a blank and assemble a set of distractor options drawn randomly from
       the remaining noun pool.
    4. Limit the number of questions produced according to the requested
       parameter.

    Args:
        text: The raw text from which to generate questions.
        num_questions: Maximum number of MCQs to create.

    Returns:
        A list of MCQ objects.
    """

    ensure_nltk_data()

    # Normalise whitespace and lower‑case the text for tokenisation.
    cleaned_text = " ".join(text.split())

    # Tokenise into sentences.
    sentences = sent_tokenize(cleaned_text)

    # Build a global noun pool from the entire document.
    tokens = word_tokenize(cleaned_text)
    pos_tags = pos_tag(tokens)
    stop_words = set(stopwords.words("english"))
    # Keep only nouns (common and proper) that are alphabetic and not stopwords.
    noun_pool = [word for word, tag in pos_tags if tag.startswith("NN") and word.lower() not in stop_words and word.isalpha()]
    # De‑duplicate while preserving order.
    seen = set()
    unique_nouns = []
    for noun in noun_pool:
        lower_n = noun.lower()
        if lower_n not in seen:
            seen.add(lower_n)
            unique_nouns.append(lower_n)

    # Guard against insufficient vocabulary.
    if not unique_nouns:
        return []

    mcq_list: List[MCQ] = []
    question_id = 0

    for sentence in sentences:
        if len(mcq_list) >= num_questions:
            break
        # Tokenise and tag the sentence.
        words = word_tokenize(sentence)
        tags = pos_tag(words)
        answer_word = None
        # Select the first eligible noun in the sentence.
        for word, tag in tags:
            if tag.startswith("NN") and word.lower() in unique_nouns:
                answer_word = word
                break
        if not answer_word:
            continue  # skip sentences without nouns
        # Construct the blanked question text.
        # Replace only the first occurrence of the answer word to avoid
        # inadvertently blanking other similar words.
        blank = "_____"
        # Use split/join to replace the first occurrence in a case‑sensitive manner.
        parts = sentence.split(answer_word, 1)
        question_text = f"{parts[0]}{blank}{parts[1]}"

        # Prepare distractor options from other nouns.
        distractor_pool = [n for n in unique_nouns if n != answer_word.lower()]
        # If fewer than 3 distractors exist, sample accordingly.
        num_distractors = min(3, len(distractor_pool))
        distractors = random.sample(distractor_pool, num_distractors) if distractor_pool else []
        # Combine correct answer (original casing) with distractors.
        options = distractors + [answer_word]
        random.shuffle(options)

        mcq = MCQ(id=question_id, question=question_text, options=options, answer=answer_word)
        mcq_list.append(mcq)
        question_id += 1

    return mcq_list


@app.route("/", methods=["GET"])
def index():
    """Render the home page with the PDF upload form."""
    return render_template("index.html")


@app.route("/quiz", methods=["POST"])
def quiz():
    """Handle PDF upload and generate a quiz from its content."""
    file = request.files.get("pdf_file")
    num_questions = request.form.get("num_questions", type=int, default=5)
    # Validate input.
    if not file or file.filename == "":
        flash("Please select a PDF file to upload.")
        return redirect(url_for("index"))
    try:
        # Read file contents into memory for PyPDF2.
        file_stream = io.BytesIO(file.read())
        text = extract_text_from_pdf(file_stream)
        mcqs = generate_mcq_questions(text, num_questions=num_questions)
        if not mcqs:
            flash("Unable to generate questions from the provided PDF. The document may not contain enough nouns.")
            return redirect(url_for("index"))
        return render_template("quiz.html", mcqs=mcqs)
    except Exception as exc:
        flash(f"An error occurred while processing the PDF: {exc}")
        return redirect(url_for("index"))


@app.route("/submit", methods=["POST"])
def submit():
    """Grade the quiz responses and display results."""
    # To grade the quiz we expect hidden fields with correct answers and
    # radio selections keyed by question id.
    results = []
    score = 0
    # We'll iterate over posted form items to reconstruct MCQs.
    # The form uses keys like q0, q1 for user answers and answer0, answer1 for correct answers.
    for key in request.form:
        if key.startswith("q") and not key.startswith("answer"):
            qid = key[1:]  # remove 'q'
            user_answer = request.form.get(key)
            correct_key = f"answer{qid}"
            correct_answer = request.form.get(correct_key)
            # Also store the original question text if provided.
            question_text = request.form.get(f"question{qid}")
            is_correct = (user_answer == correct_answer)
            if is_correct:
                score += 1
            results.append({
                "question": question_text or "",
                "user_answer": user_answer,
                "correct": correct_answer,
                "is_correct": is_correct,
            })
    return render_template("result.html", results=results, score=score)


if __name__ == "__main__":
    # When running locally for testing, enable debug mode.
    app.run(debug=True)