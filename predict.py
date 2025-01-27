import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('target')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    target = y_pred >= 0.5

    result = {
        'credit_score_probability': float(y_pred),
        'credit_score': bool(target)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8885)
