import json
from concurrent.futures import ThreadPoolExecutor

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

OUTPUT_FILENAME = "reasoning_trace_tokens.json"
WORKERS = 48

LEMMATIZER = WordNetLemmatizer()


def inputs(context):
    collated_path = context.get("annotator_path")
    if not collated_path or not collated_path.exists():
        raise FileNotFoundError("annotator output missing; run with --jobs annotator first.")
    return {
        "collated_path": collated_path,
        "output_root": context["output_root"],
    }


def outputs(context):
    return {"token_sequences_path": context["output_root"] / OUTPUT_FILENAME}


def _process_trace(trace, lemmatizer, stopword_set):
    tokens = nltk.word_tokenize(trace["reasoning_trace"])
    normalized = []
    for token in tokens:
        lemma = lemmatizer.lemmatize(token.strip().lower())
        if lemma in stopword_set:
            continue
        if not any(ch.isalnum() for ch in lemma):
            continue
        normalized.append(lemma)
    return {
        "pmcid": trace.get("pmcid"),
        "model": trace.get("model"),
        "token_sequence": normalized,
    }


def run(collated_path, output_root):
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    wordnet.ensure_loaded()

    data = json.load(collated_path.open())
    lemmatizer = WordNetLemmatizer()
    stopword_set = set(stopwords.words("english"))

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        rows = list(executor.map(lambda tr: _process_trace(tr, lemmatizer, stopword_set), data["traces"]))

    output_path = output_root / OUTPUT_FILENAME
    payload = {"traces": rows}
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote token sequences to {output_path}")
