from flask import Flask, request, jsonify
import pickle
import pandas as pd
from collections import Counter

# Load models and vectorizer
LR = pickle.load(open('lr_model.pkl', 'rb'))
DT = pickle.load(open('dt_model.pkl', 'rb'))
GBC = pickle.load(open('gbc_model.pkl', 'rb'))
RFC = pickle.load(open('rfc_model.pkl', 'rb'))
vectorization = pickle.load(open('vectorizer.pkl', 'rb'))

# Define your preprocessing and label functions
def wordopt(text):
    # Add your preprocessing logic here
    return text.lower()  # Simplified for example

def output_label(pred):
    return "Fake" if pred == 0 else "Real"

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news = data['news']

    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])

    pred_LR = LR.predict(new_xv_test)[0]
    pred_DT = DT.predict(new_xv_test)[0]
    pred_GBC = GBC.predict(new_xv_test)[0]
    pred_RFC = RFC.predict(new_xv_test)[0]

    predictions = [pred_LR, pred_DT, pred_GBC, pred_RFC]

    prob_LR = LR.predict_proba(new_xv_test)[0][pred_LR] * 100
    prob_DT = DT.predict_proba(new_xv_test)[0][pred_DT] * 100
    prob_GBC = GBC.predict_proba(new_xv_test)[0][pred_GBC] * 100
    prob_RFC = RFC.predict_proba(new_xv_test)[0][pred_RFC] * 100

    ensemble_pred = Counter(predictions).most_common(1)[0][0]
    ensemble_confidence = sum([
        prob_LR if pred_LR == ensemble_pred else 0,
        prob_DT if pred_DT == ensemble_pred else 0,
        prob_GBC if pred_GBC == ensemble_pred else 0,
        prob_RFC if pred_RFC == ensemble_pred else 0,
    ]) / predictions.count(ensemble_pred)

    result = {
        "ensemble_prediction": output_label(ensemble_pred),
        "confidence": round(ensemble_confidence, 2),
        "individual_predictions": {
            "LR": output_label(pred_LR),
            "DT": output_label(pred_DT),
            "GBC": output_label(pred_GBC),
            "RFC": output_label(pred_RFC)
        },
        "individual_confidences": {
            "LR": round(prob_LR, 2),
            "DT": round(prob_DT, 2),
            "GBC": round(prob_GBC, 2),
            "RFC": round(prob_RFC, 2),
        }
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
