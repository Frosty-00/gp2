"""
predict_stance.py — Stance Detection Inference Module

Loads the fine-tuned DistilBERT stance model once at import time and exposes
a predict_stance() function that classifies the relationship between a
headline and a body text as Support / Against / Neutral.

Designed for CPU-only deployment.

Run standalone:  python -m src.predict_stance
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OpenMP conflict on Windows

import re
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. Resolve model path (relative to project root, not to this file)
#    Use pathlib.Path so from_pretrained() treats it as a local directory,
#    even when the path contains spaces (e.g. "AI APPLICATION").
# ---------------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "stance_model"

# ---------------------------------------------------------------------------
# 2. Device selection
# ---------------------------------------------------------------------------
def _select_device() -> torch.device:
    """Use GPU if available and compatible, otherwise CPU."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


DEVICE = _select_device()

# ---------------------------------------------------------------------------
# 3. Load model and tokenizer (once at import time)
# ---------------------------------------------------------------------------
print(f"[predict_stance] Loading model from {MODEL_DIR} ...")
_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
_model.to(DEVICE)
_model.eval()
print(f"[predict_stance] Model loaded on {DEVICE}")

# ---------------------------------------------------------------------------
# 4. Label mapping (must match training)
# ---------------------------------------------------------------------------
ID_TO_LABEL = {0: "Support", 1: "Against", 2: "Neutral"}
MAX_LENGTH = 256

# ---------------------------------------------------------------------------
# 5. Text cleaning (same as training)
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Basic cleaning: strip + collapse whitespace."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------
VOWELS = set("aeiouy")

# Headline thresholds (lighter — headlines are naturally short)
HL_MIN_CHARS = 10          # at least 10 characters
HL_MIN_WORDS = 3           # at least 3 words
HL_MIN_ALPHA_RATIO = 0.50  # at least 50% letters
HL_MAX_WORD_LEN = 25       # single word length cap

# Body thresholds (stricter — bodies should be real text)
BD_MIN_CHARS = 30          # at least 30 characters
BD_MIN_WORDS = 6           # at least 6 words
BD_MIN_ALPHA_RATIO = 0.50  # at least 50% letters
BD_MAX_REPEAT_RATIO = 0.60 # reject if one char dominates > 60%
BD_MAX_WORD_LEN = 25       # single word length cap
BD_MIN_REAL_WORD_RATIO = 0.30  # at least 30% of words must be real

INVALID_RESULT = {
    "label": "Invalid Input",
    "confidence": 0.0,
    "message": "Please enter a meaningful headline and article body.",
}


def _is_real_word(word: str) -> bool:
    """
    Check if a word is likely real English using two heuristics:
    1. Vowel check — real words almost always contain a vowel.
    2. Tokenizer check — real words produce 1 token; gibberish produces many.
    """
    w = word.lower().strip(".,!?;:\"'()-")
    if not w:
        return False

    # short words and pure numbers are fine
    if len(w) <= 2 or w.isdigit():
        return True

    # unreasonably long → gibberish
    if len(w) > BD_MAX_WORD_LEN:
        return False

    # vowel check
    if not any(c in VOWELS for c in w) and len(w) >= 4:
        return False

    # tokenizer subword check
    token_ids = _tokenizer.encode(w, add_special_tokens=False)
    max_tokens = 2 if len(w) >= 7 else 1
    return len(token_ids) <= max_tokens


def _validate_headline(text: str) -> str | None:
    """
    Light validation for the headline.
    Returns an error message if invalid, or None if OK.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "Headline is empty."

    text = text.strip()

    if len(text) < HL_MIN_CHARS:
        return f"Headline is too short (minimum {HL_MIN_CHARS} characters)."

    words = text.split()
    if len(words) < HL_MIN_WORDS:
        return f"Headline has too few words (minimum {HL_MIN_WORDS} words)."

    # letter ratio
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count / len(text) < HL_MIN_ALPHA_RATIO:
        return "Headline does not look like natural language."

    # any single word too long → gibberish
    if any(len(w) > HL_MAX_WORD_LEN for w in words):
        return "Headline contains unrecognizable words."

    return None


def _validate_body(text: str) -> str | None:
    """
    Stricter validation for the article body.
    Returns an error message if invalid, or None if OK.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "Article body is empty."

    text = text.strip()

    if len(text) < BD_MIN_CHARS:
        return f"Article body is too short (minimum {BD_MIN_CHARS} characters)."

    words = text.split()
    if len(words) < BD_MIN_WORDS:
        return f"Article body has too few words (minimum {BD_MIN_WORDS} words)."

    # letter ratio
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count / len(text) < BD_MIN_ALPHA_RATIO:
        return "Article body does not look like natural language."

    # single-character spam
    char_counts = Counter(text.lower().replace(" ", ""))
    if char_counts:
        most_common_ratio = char_counts.most_common(1)[0][1] / len(text.replace(" ", ""))
        if most_common_ratio > BD_MAX_REPEAT_RATIO:
            return "Article body looks like repeated or random characters."

    # very few unique characters → keyboard mash
    unique_chars = len(set(text.lower().replace(" ", "")))
    if len(text) > 40 and unique_chars < 8:
        return "Article body does not look like natural language."

    # need at least some recognizable alpha words
    alpha_words = [w for w in words if len(w) >= 3 and w.isalpha()]
    if len(alpha_words) < 3:
        return "Article body does not contain enough recognizable words."

    # tokenizer-based real word check
    real_count = sum(1 for w in alpha_words if _is_real_word(w))
    if alpha_words and real_count / len(alpha_words) < BD_MIN_REAL_WORD_RATIO:
        return "Article body contains too many unrecognizable words."

    return None


# ---------------------------------------------------------------------------
# 7. Main inference function
# ---------------------------------------------------------------------------
def predict_stance(headline: str, body: str) -> dict:
    """
    Predict the stance of a body text toward a headline.

    Args:
        headline: The claim or news headline.
        body:     The article body text.

    Returns:
        {
            "label": "Support" | "Against" | "Neutral",
            "confidence": float  (0.0 – 1.0)
        }
        or {"label": "Invalid Input", ...} when validation fails.
    """
    # --- Validate headline (light) ---
    hl_error = _validate_headline(headline)
    if hl_error:
        return {**INVALID_RESULT, "message": hl_error}

    # --- Validate body (strict) ---
    bd_error = _validate_body(body)
    if bd_error:
        return {**INVALID_RESULT, "message": bd_error}

    # --- Combine headline + body (same format as training) ---
    combined = _clean_text(headline) + " [SEP] " + _clean_text(body)

    # --- Tokenize ---
    inputs = _tokenizer(
        combined,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze()
    pred_id = torch.argmax(probs).item()
    confidence = probs[pred_id].item()

    return {
        "label": ID_TO_LABEL[pred_id],
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# 8. Quick test (run with: python -m src.predict_stance)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    examples = [
        {
            "headline": "New study confirms vaccines are safe and effective",
            "body": (
                "Researchers at Johns Hopkins University published findings "
                "that confirm the safety and efficacy of current vaccines. "
                "The study reviewed data from over 10 million patients."
            ),
            "expected": "Support",
        },
        {
            "headline": "City to ban all cars from downtown area",
            "body": (
                "The mayor's office denied rumors of a car ban. Officials "
                "stated that no such policy is under consideration and "
                "traffic will continue as normal."
            ),
            "expected": "Against",
        },
        {
            "headline": "Tech company announces record profits",
            "body": (
                "A new species of frog was discovered in the Amazon "
                "rainforest by a team of biologists from Brazil. The frog "
                "has a unique blue coloration."
            ),
            "expected": "Neutral",
        },
        # --- Validation test cases ---
        {
            "headline": "ab",
            "body": "This is a valid article body with enough words to pass validation.",
            "expected": "Invalid Input",
        },
        {
            "headline": "A valid headline for testing purposes",
            "body": "asdfghjkl qwerty zxcvbn poiuytr lkjhgf mnbvcx woeiru asdfer",
            "expected": "Invalid Input",
        },
        {
            "headline": "qwert asdf zxcvb",
            "body": "This is a valid article body with enough words to pass validation.",
            "expected": "Invalid Input",
        },
    ]

    print("=" * 65)
    print("  Stance Detection — Quick Test")
    print("=" * 65)
    print(f"  Device: {DEVICE}")
    print(f"  Model : {MODEL_DIR}")
    print("=" * 65)

    for ex in examples:
        result = predict_stance(ex["headline"], ex["body"])
        match = "✓" if result["label"] == ex["expected"] else "✗"
        print(f"\n  Headline : {ex['headline'][:55]}")
        print(f"  Expected : {ex['expected']}")
        print(f"  Predicted: {result['label']}  ", end="")
        if result["label"] == "Invalid Input":
            print(f"({result['message']})  {match}")
        else:
            print(f"(confidence: {result['confidence']:.1%})  {match}")

    print("\n" + "=" * 65)
