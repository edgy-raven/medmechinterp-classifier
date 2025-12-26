from boltons.iterutils import chunked
import atexit
import json
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
from openai import OpenAI
import tiktoken


EMBEDDING_BATCH_SIZE = 1028
EMBEDDING_MODEL = "text-embedding-3-small"
TOKEN_ENCODING = tiktoken.encoding_for_model(EMBEDDING_MODEL)

_CLIENT = None
_EMBED_CACHE = {}
_CACHE_LOADED = False
CACHE_PATH = Path("output_artifacts") / "embedding_cache.pkl"
_SIM_CACHE = {}


def initialize_client(api_key=None):
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=api_key)
        _load_cache()
    return _CLIENT


def _load_cache():
    global _CACHE_LOADED, _EMBED_CACHE
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    if not CACHE_PATH.exists():
        return
    with CACHE_PATH.open("rb") as f:
        _EMBED_CACHE = pickle.load(f)


def _dump_cache():
    if not _EMBED_CACHE:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as f:
        pickle.dump(_EMBED_CACHE, f)


atexit.register(_dump_cache)


def embed_texts(texts):
    def is_blank(text):
        if text is None:
            return True
        if isinstance(text, str):
            return text.strip() == ""
        return False

    to_embed = []
    seen = set()
    for text in texts:
        if is_blank(text):
            continue
        if text in _EMBED_CACHE or text in seen:
            continue
        seen.add(text)
        to_embed.append(text)

    for batch in chunked(to_embed, EMBEDDING_BATCH_SIZE):
        while True:
            try:
                res = initialize_client().embeddings.create(model=EMBEDDING_MODEL, input=batch)
                break
            except Exception as exc:
                exc_name = exc.__class__.__name__
                if exc_name == "RateLimitError":
                    time.sleep(0.02 + random.random() * 0.02)
                    continue
                if exc_name == "BadRequestError":
                    payload = {"model": EMBEDDING_MODEL, "input": batch}
                    try:
                        dumped = json.dumps(payload, indent=2, ensure_ascii=True)
                    except TypeError:
                        dumped = str(payload)
                    sys.stderr.write("BadRequestError from OpenAI embeddings request. Payload:\n")
                    sys.stderr.write(dumped + "\n")
                raise
        for key, item in zip(batch, res.data):
            _EMBED_CACHE[key] = np.array(item.embedding, dtype=float)

    dim = None
    for text in texts:
        if is_blank(text):
            continue
        emb = _EMBED_CACHE.get(text)
        if emb is not None:
            dim = len(emb)
            break
    if dim is None:
        dim = 0

    empty_emb = np.zeros(dim, dtype=float)
    results = []
    for text in texts:
        if is_blank(text):
            results.append(empty_emb.copy())
        else:
            results.append(_EMBED_CACHE[text])
    return results


def similarity(token_a, token_b):
    if token_a == token_b:
        return 1.0
    key = (token_a, token_b) if token_a <= token_b else (token_b, token_a)
    if key in _SIM_CACHE:
        return _SIM_CACHE[key]
    emb_a, emb_b = embed_texts([token_a, token_b])
    sim = float(np.dot(emb_a, emb_b))
    _SIM_CACHE[key] = sim
    return sim
