import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow as tf
import random
import json
import numpy as np
import joblib
import warnings
from flask import Flask, request, jsonify
import nltk
nltk.download('punkt')

app = Flask(__name__)

app = Flask(__name__)

# Load data và mô hình cho bot chat
stemmer = LancasterStemmer()

with open('intents.json') as f:
    data = json.load(f)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chatbot_response(inp):
    results = model.predict([bag_of_words(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            return random.choice(responses)

@app.route('/chat', methods=['GET'])
def chat():
    user_message = request.args.get('message')
    bot_response = chatbot_response(user_message)
    return jsonify({"response": bot_response})



warnings.filterwarnings("ignore", category=UserWarning)

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
