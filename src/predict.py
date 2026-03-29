"""
predict.py — Fake News Inference Module

Loads the fine-tuned DistilBERT model once at import time and exposes
a predict_text() function that returns a label ("Fake" / "Real") with
a confidence score.  Designed for CPU-only deployment.
"""

import os
import re
from collections import Counter
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OpenMP conflict on Windows

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. Resolve model path (relative to project root, not to this file)
#    Use pathlib.Path so from_pretrained() treats it as a local directory,
#    even when the path contains spaces (e.g. "AI APPLICATION").
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "fake_news_model"

# ---------------------------------------------------------------------------
# 2. Device selection — use GPU if available *and* compatible, otherwise CPU
# ---------------------------------------------------------------------------
def _select_device() -> torch.device:
    """Pick CUDA only when the current GPU is actually usable by PyTorch."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")

device = _select_device()

# ---------------------------------------------------------------------------
# 3. Load model & tokenizer once (runs at import time)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
model.to(device)
model.eval()  # switch to inference mode (disables dropout)

# ---------------------------------------------------------------------------
# 4. Label mapping (must match training labels)
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "Real", 1: "Fake"}

# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------
MIN_CHARS = 30          # minimum character count
MIN_WORDS = 6           # minimum word count
MIN_ALPHA_RATIO = 0.50  # at least 50% of characters should be letters
MAX_REPEAT_RATIO = 0.60 # reject if any single character is >60% of text
MAX_WORD_LEN = 25       # words longer than this are likely gibberish
MIN_REAL_WORD_RATIO = 0.30  # at least 30% of words must be "real" words

INVALID_RESULT = {
    "label": "Invalid Input",
    "confidence": 0.0,
    "message": "Please enter a meaningful news-like sentence or paragraph.",
}


VOWELS = set("aeiouy")


def _is_real_word(word: str) -> bool:
    """
    Check if a word is likely real English using two heuristics:

    1. Tokenizer check — real words are recognized by DistilBERT as 1 token.
       Gibberish like "asdfghjkl" gets split into many subword pieces.
    2. Vowel check — real English words almost always contain a vowel.
       Gibberish fragments like "fasd", "dfa", "adf" often don't follow
       natural vowel-consonant patterns.

    Also accepts common patterns: numbers, short words, abbreviations.
    """
    w = word.lower().strip(".,!?;:\"'()-")
    if not w:
        return False

    # short words (1-2 chars) and pure numbers are fine
    if len(w) <= 2 or w.isdigit():
        return True

    # if the word is unreasonably long, it's gibberish
    if len(w) > MAX_WORD_LEN:
        return False

    # vowel check: real English words need at least one vowel
    # exceptions are rare abbreviations (e.g. "mrs", "dr") — they are short
    has_vowel = any(c in VOWELS for c in w)
    if not has_vowel and len(w) >= 4:
        return False

    # tokenizer check: real words produce 1 token, gibberish produces many
    # e.g. "president" → ["president"]  (1 token)
    #      "asdfghjkl" → ["as","##df","##gh","##jk","##l"] (5 tokens)
    token_ids = tokenizer.encode(w, add_special_tokens=False)

    # strict: only 1 token = definitely a known word
    # for longer words (7+ chars), allow up to 2 tokens (e.g. compound words)
    max_tokens = 2 if len(w) >= 7 else 1
    return len(token_ids) <= max_tokens


def validate_input(text: str) -> str | None:
    """
    Check whether text looks like a real news snippet.
    Returns an error message string if invalid, or None if OK.
    """
    # empty / wrong type
    if not text or not isinstance(text, str) or text.strip() == "":
        return "Input is empty."

    text = text.strip()

    # too short
    if len(text) < MIN_CHARS:
        return f"Input is too short (minimum {MIN_CHARS} characters)."

    # too few words
    words = text.split()
    if len(words) < MIN_WORDS:
        return f"Input has too few words (minimum {MIN_WORDS} words)."

    # low letter ratio → random characters / numbers / symbols
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count / len(text) < MIN_ALPHA_RATIO:
        return "Input does not look like natural language text."

    # single-character spam (e.g. "aaaaaaaaaa")
    char_counts = Counter(text.lower().replace(" ", ""))
    if char_counts:
        most_common_ratio = char_counts.most_common(1)[0][1] / len(text.replace(" ", ""))
        if most_common_ratio > MAX_REPEAT_RATIO:
            return "Input looks like repeated or random characters."

    # very few unique characters → keyboard mash
    unique_chars = len(set(text.lower().replace(" ", "")))
    if len(text) > 40 and unique_chars < 8:
        return "Input does not look like natural language text."

    # need at least some recognizable words (3+ letters)
    alpha_words = [w for w in words if len(w) >= 3 and w.isalpha()]
    if len(alpha_words) < 3:
        return "Input does not contain enough recognizable words."

    # ---- NEW: check if words are real English (tokenizer-based) ----
    # only check alphabetic words with 3+ characters (skip numbers, punctuation)
    real_count = sum(1 for w in alpha_words if _is_real_word(w))
    if alpha_words and real_count / len(alpha_words) < MIN_REAL_WORD_RATIO:
        return "Input contains too many unrecognizable words — please enter real text."

    return None  # all checks passed


# ---------------------------------------------------------------------------
# 6. Prediction function
# ---------------------------------------------------------------------------
def predict_text(text: str) -> dict:
    """
    Classify a single news article as Fake or Real.

    Parameters
    ----------
    text : str
        The news article (title + body, or any free text).

    Returns
    -------
    dict
        {"label": ..., "confidence": float}
        Includes "message" key when input is invalid.
    """
    # --- validate input before running the model ---
    error = validate_input(text)
    if error:
        return {**INVALID_RESULT, "message": error}

    # --- tokenize (same settings used during training) ---
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",      # return PyTorch tensors
    )

    # move tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # --- run inference (no gradient computation needed) ---
    with torch.no_grad():
        outputs = model(**inputs)

    # --- convert logits → probabilities ---
    probs = F.softmax(outputs.logits, dim=1)   # shape: (1, 2)

    predicted_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_id].item()

    return {
        "label": LABEL_MAP[predicted_id],
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# 7. Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model loaded from : {MODEL_DIR}")
    print(f"Device            : {device}\n")

    examples = [
        "BREAKING: Scientists confirm the moon is made of cheese, NASA covers it up.",
        (
            "WASHINGTON (Reuters) - The U.S. Senate passed a bipartisan infrastructure "
            "bill on Tuesday, sending $550 billion in new federal spending to roads, "
            "bridges and broadband internet."
        ),
        "You won't BELIEVE what this celebrity did — doctors are SHOCKED!",
        "asdfghjkl asdfghjkl asdfghjkl asdfghjkl asdfghjkl asdfghjkl",
        "hello",
        "",
    ]

    for i, text in enumerate(examples, 1):
        result = predict_text(text)
        display = text[:80] if text else "(empty)"
        print(f"Example {i}")
        print(f"  Text       : {display}")
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.2%}")
        if "message" in result:
            print(f"  Message    : {result['message']}")
        print()
