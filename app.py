"""Flask app: predict whether a respondent is a regular online shopper.

Run:  python app.py   ->  http://127.0.0.1:5000
Needs shopping_model.joblib in the same folder (created by the notebook).
"""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

MODEL_PATH = Path(__file__).parent / "shopping_model.joblib"
model = joblib.load(MODEL_PATH)

# Read the fields and allowed answers straight from the trained encoder,
# so the form always matches the model.
encoder = model.named_steps["pre"].named_transformers_["oh"]
FIELDS = dict(zip(encoder.feature_names_in_, [list(c) for c in encoder.categories_]))

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Online Shopper Predictor</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#222}
 h1{color:#1f3d5c} label{display:block;margin-top:.9rem;font-weight:600;font-size:.9rem}
 select{width:100%;padding:.45rem;margin-top:.2rem}
 button{margin-top:1.4rem;padding:.7rem 1.4rem;background:#1f3d5c;color:#fff;border:0;border-radius:6px;font-size:1rem;cursor:pointer}
 .result{margin-top:1.5rem;padding:1rem;border-radius:8px;background:#eef4fa;border-left:5px solid #1f3d5c}
</style></head><body>
<h1>Online Shopper Predictor</h1>
<p>Fill in the survey answers to predict if someone shops online regularly (monthly or more).</p>
<form method="post">
{% for name, options in fields.items() %}
 <label>{{ name.replace('_',' ') }}
  <select name="{{ name }}">
  {% for o in options %}<option value="{{ o }}" {{ 'selected' if values.get(name)==o }}>{{ o }}</option>{% endfor %}
  </select></label>
{% endfor %}
 <button type="submit">Predict</button>
</form>
{% if result %}
<div class="result"><b>{{ result.label }}</b><br>Probability of being a regular shopper: {{ result.probability }}%</div>
{% endif %}
</body></html>
"""


def predict(answers: dict) -> dict:
    missing = [f for f in FIELDS if f not in answers]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    row = pd.DataFrame([{f: answers[f] for f in FIELDS}])
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "regular_shopper": prob >= 0.5,
        "label": "Likely a regular online shopper" if prob >= 0.5 else "Likely a rare/non online shopper",
        "probability": round(prob * 100, 1),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, values = None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            result = predict(values)
        except ValueError as e:
            result = {"label": str(e), "probability": "-"}
    return render_template_string(PAGE, fields=FIELDS, values=values, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict(data))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.route("/api/fields")
def api_fields():
    return jsonify(FIELDS)


if __name__ == "__main__":
    app.run(debug=True)
