import os.path
import re

from typing import Any, Dict, List, Optional

from langdetect import detect, DetectorFactory
from readabilipy import simple_json_from_html_string
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load
from sklearn.metrics import f1_score

DetectorFactory.seed = 0

ENCODING_DECL_REGEX = re.compile(r"<\?xml[^>]*encoding=[^>]*\?>", re.IGNORECASE)


def get_language(text: str) -> str:
    return detect(text)


def smart_truncate(
    content,
    length=100,
    suffix="...",
):
    if len(content) <= length:
        return content
    else:
        return " ".join(content[: length + 1].split(" ")[0:-1]) + suffix


def get_page_raw_text_content(html_content: str) -> str:
    if not html_content or not html_content.strip():
        return ""

    # Remove XML encoding declarations that trigger lxml ValueError
    clean_html_str = ENCODING_DECL_REGEX.sub("", html_content)

    try:
        html_plain_text_dict: Optional[Dict[str, Any]] = simple_json_from_html_string(
            clean_html_str,
            use_readability=False,
        )["plain_text"]

        if not html_plain_text_dict:
            return ""

        text: str = ". ".join(t["text"] for t in html_plain_text_dict)

    except (IndexError, ValueError):
        print("--> Error parsing HTML content! Blank response or malformed HTML.")
        return ""

    return text


def test_svm_classifier(
    clf: svm.LinearSVC,
    vectorizer: TfidfVectorizer,
    corpus: list[str],
    labels: list[int],
):

    samples_features = vectorizer.transform(corpus)
    predicted_labels = clf.predict(samples_features)
    score = f1_score(labels, list(predicted_labels))

    return score


def save_model_to_disk(
    model,
    save_dir,
    model_name,
):
    dump(
        model,
        os.path.join(save_dir, f"models/{model_name}.joblib"),
    )


def train_svm_classifier(
    corpus: list[str],
    labels: list[int],
    save_vectorizer: bool = True,
    save_model: bool = True,
    *,
    C: float = 1000.0,
    max_iter: int = 100000,
):

    vectorizer_name = "models/svm_tfidf_vectorizer.joblib"
    current_dir = os.path.dirname(__file__)

    if save_vectorizer:
        vectorizer = TfidfVectorizer()
        samples_features = vectorizer.fit_transform(corpus)

        dump(
            vectorizer,
            os.path.join(
                current_dir,
                vectorizer_name,
            ),
        )

    else:
        vectorizer = load_trained_model("svm_tfidf_vectorizer.joblib")
        samples_features = vectorizer.transform(corpus)

    clf = svm.LinearSVC(
        C=C,
        max_iter=max_iter,
    )
    clf.fit(
        samples_features,
        labels,
    )

    if save_model:
        model_name: str = f"svm_linear_classifier_C_{C}_max_iter_{max_iter}"
        save_model_to_disk(
            clf,
            current_dir,
            model_name,
        )

    return clf


def train_naive_bayes_classifier(
    corpus: List[str],
    labels: List[int],
    save_vectorizer: bool = True,
):
    vectorizer_name = "models/svm_tfidf_vectorizer.joblib"
    current_dir = os.path.dirname(__file__)

    if save_vectorizer:
        vectorizer = TfidfVectorizer()
        samples_features = vectorizer.fit_transform(corpus)
        dump(
            vectorizer,
            os.path.join(
                current_dir,
                vectorizer_name,
            ),
        )

    else:
        vectorizer = load_trained_model("svm_tfidf_vectorizer.joblib")
        samples_features = vectorizer.transform(corpus)

    clf = MultinomialNB(force_alpha=True)
    clf.fit(samples_features, labels)

    dump(
        clf,
        os.path.join(
            current_dir,
            "models/multinomial_naive_bayes_classifier.joblib",
        ),
    )


def load_trained_model(model_file_name: str):
    current_dir = os.path.dirname(__file__)
    model = load(
        os.path.join(current_dir, f"models/{model_file_name}"),
    )

    return model


_MODEL_CACHE = {}

def get_cached_model(filename: str):
    if filename not in _MODEL_CACHE:
        _MODEL_CACHE[filename] = load_trained_model(filename)
    return _MODEL_CACHE[filename]
