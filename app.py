"""
app.py — Flask Web Application for Fake News & Stance Detection

Serves a web interface with two modules:
  1. Fake News Detection  — paste an article → Fake / Real
  2. Stance Detection      — enter headline + body → Support / Against / Neutral

Run:  python app.py
Open: http://localhost:5000
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix OpenMP conflict on Windows

import sys
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# Make sure Python can find the src/ package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_text              # fake news model
from src.predict_stance import predict_stance      # stance model

# ---------------------------------------------------------------------------
# Create Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Preset demo examples (used in "Lite Demo Examples" mode)
# These are just sample inputs — the model is always the same full model.
# ---------------------------------------------------------------------------
DEMO_EXAMPLES = {
    "real_clear": {
        "name": "Clear Real News (Reuters)",
        "text": (
            "ANKARA (Reuters) - A Turkish mayor announced on Monday he had quit "
            "his post and left President Tayyip Erdogan's ruling AK Party after "
            "pressure and threats \"beyond unbearable\", becoming the sixth mayor "
            "in recent weeks to fall victim of a purge of local government."
        ),
    },
    "fake_clear": {
        "name": "Clear Fake News (clickbait)",
        "text": (
            "Donald Trump Humiliates Chris Christie AGAIN And It's Worse Than "
            "Ever Before. Chris Christie's strange loyalty to presumptive "
            "Republican nominee Donald Trump has been questioned since the day "
            "he pledged his unwavering support to the business mogul. However, "
            "every time Trump insults or embarrasses the New Jersey governor, "
            "Christie loses even more of the public's respect."
        ),
    },
    "borderline": {
        "name": "Borderline / Uncertain",
        "text": (
            "Sources close to the administration say a major policy announcement "
            "is expected sometime next week, though officials have declined to "
            "confirm the details. The move could reshape federal regulations on "
            "technology companies."
        ),
    },
    "rewritten": {
        "name": "Rewritten Formal Style",
        "text": (
            "The United States Senate convened on Tuesday to deliberate on a "
            "proposed infrastructure spending package. The legislation, which "
            "allocates approximately $550 billion in new federal expenditure, "
            "targets improvements to roads, bridges, and broadband internet "
            "access across the nation."
        ),
    },
    "gibberish": {
        "name": "Invalid / Gibberish Input",
        "text": "asdfghjkl qwerty zxcvbn poiuytr lkjhgf mnbvcx woeiru",
    },
}


# ---------------------------------------------------------------------------
# Preset stance demo examples (headline + body pairs)
# ---------------------------------------------------------------------------
STANCE_DEMOS = {
    "support": {
        "name": "Support — body agrees with headline",
        "headline": "New study confirms vaccines are safe and effective",
        "body": (
            "Researchers at Johns Hopkins University published findings that "
            "confirm the safety and efficacy of current vaccines. The study "
            "reviewed data from over 10 million patients and found no "
            "significant adverse effects."
        ),
    },
    "against": {
        "name": "Against — body contradicts headline",
        "headline": "City to ban all cars from downtown area",
        "body": (
            "The mayor's office denied rumors of a car ban. Officials stated "
            "that no such policy is under consideration and traffic will "
            "continue as normal in the downtown district."
        ),
    },
    "neutral": {
        "name": "Neutral — body is unrelated to headline",
        "headline": "Tech company announces record profits",
        "body": (
            "A new species of frog was discovered in the Amazon rainforest "
            "by a team of biologists from Brazil. The frog has a unique "
            "blue coloration not seen in any other known amphibian species."
        ),
    },
    "borderline": {
        "name": "Borderline — body discusses but takes no clear side",
        "headline": "Government plans to raise minimum wage",
        "body": (
            "Economists are divided on the proposed minimum wage increase. "
            "Some argue it will boost consumer spending, while others warn "
            "it could lead to job losses in small businesses. The debate "
            "is expected to continue through the legislative session."
        ),
    },
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    """Render the main page (no result yet)."""
    return render_template("index.html", demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept user input, run prediction, and return the result.
    Supports both form submission and JSON API calls.
    """
    # Get text from form data or JSON body
    text = request.form.get("text", "").strip()
    if not text and request.is_json:
        text = request.get_json().get("text", "").strip()

    # Handle empty input
    if not text:
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            error="Please paste a news article before clicking Analyze.",
        )

    # Run prediction
    try:
        result = predict_text(text)
    except Exception as e:
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            text=text,
            error=f"Prediction failed: {e}",
        )

    # If validation rejected the input, show a warning instead of Fake/Real
    if result["label"] == "Invalid Input":
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            text=text,
            error=result.get("message", "Invalid input."),
        )

    return render_template(
        "index.html",
        demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
        text=text,
        label=result["label"],
        confidence=result["confidence"],
    )


# ---------------------------------------------------------------------------
# Stance Detection route
# ---------------------------------------------------------------------------
@app.route("/predict_stance", methods=["POST"])
def stance():
    """Accept headline + body, run stance prediction, return result."""
    headline = request.form.get("headline", "").strip()
    body = request.form.get("body", "").strip()

    # Handle empty input
    if not headline and not body:
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            stance_error="Please enter both a headline and article body.",
        )

    if not headline:
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            stance_body=body,
            stance_error="Please enter a headline.",
        )

    if not body:
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            stance_headline=headline,
            stance_error="Please enter the article body.",
        )

    # Run prediction
    try:
        result = predict_stance(headline, body)
    except Exception as e:
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            stance_headline=headline,
            stance_body=body,
            stance_error=f"Prediction failed: {e}",
        )

    # If validation rejected the input, show a warning instead of stance result
    if result["label"] == "Invalid Input":
        return render_template(
            "index.html",
            demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
            stance_headline=headline,
            stance_body=body,
            stance_error=result.get("message", "Invalid input."),
        )

    return render_template(
        "index.html",
        demos=DEMO_EXAMPLES, stance_demos=STANCE_DEMOS,
        stance_headline=headline,
        stance_body=body,
        stance_label=result["label"],
        stance_confidence=result["confidence"],
    )


# ---------------------------------------------------------------------------
# Start the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  Fake News & Stance Detector")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=False, host="127.0.0.1", port=5000)
