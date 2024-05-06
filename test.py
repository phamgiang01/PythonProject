
import random
import json
import numpy as np
import joblib
import warnings
from flask import Flask, request, jsonify

app = Flask(__name__)

app = Flask(__name__)


# Loading the trained Random Forest model and related data
model_data = joblib.load('random_forest_model_data.joblib')
loaded_model = model_data['model']
symptoms = model_data['X_columns']
encoder = model_data['encoder']

# Symptoms prediction function
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom_index[value] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}



def predictCertainDiseases(symptoms):
    symptoms = symptoms.split(",")

    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            return jsonify({"error": f"Symptom '{symptom}' not recognized."})

    # Reshaping the input data and converting it
    # into a suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # Generating probability outputs for each class
    rf_probabilities = loaded_model.predict_proba(input_data)[0]

    # Checking if any disease has a probability greater than 0.1
    certain_predictions = [
        data_dict["predictions_classes"][i]
        for i, prob in enumerate(rf_probabilities)
        if prob > 0.1
    ]

    if certain_predictions:
        # Returning the list of certain predicted diseases
        return {"certain_predicted_diseases": [disease for disease in certain_predictions]}
    else:
        return {"message": "No certain prediction for any disease."}

@app.route('/predict_diseases', methods=['GET', 'POST'])
def predict_diseases():
    try:
        if request.method == 'GET':
            symptoms = request.args.get('symptoms')
        elif request.method == 'POST':
            data = request.get_json()
            symptoms = data.get('symptoms')
        else:
            return jsonify({"error": "Invalid request method."})

        result = predictCertainDiseases(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)

