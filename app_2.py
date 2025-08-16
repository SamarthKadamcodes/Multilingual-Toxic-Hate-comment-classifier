from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import numpy as np

app = Flask(__name__)

labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]

multi_model_name = "unitary/multilingual-toxic-xlm-roberta"
multi_tokenizer = AutoTokenizer.from_pretrained(multi_model_name)
multi_model = AutoModelForSequenceClassification.from_pretrained(multi_model_name)

eng_model_name = "unitary/toxic-bert"
eng_tokenizer = AutoTokenizer.from_pretrained(eng_model_name)
eng_model = AutoModelForSequenceClassification.from_pretrained(eng_model_name)

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits).numpy()[0]
    return {label: round(float(score), 2) for label, score in zip(labels, scores)}

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    translated_text = ""
    final_result = None

    if request.method == "POST":
        text = request.form["comment"]

        result_multi = predict(text, multi_tokenizer, multi_model)

        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            result_translated = predict(translated_text, eng_tokenizer, eng_model)
        except Exception:
            translated_text = None
            result_translated = None

        if result_translated:
            if max(result_translated.values()) > max(result_multi.values()):
                final_result = result_translated
            else:
                final_result = result_multi
        else:
            final_result = result_multi

    return render_template(
        "index.html",
        text=text,
        translated_text=translated_text,
        result_multi=final_result,
        labels=labels
    )

if __name__ == "__main__":
    app.run(debug=True)