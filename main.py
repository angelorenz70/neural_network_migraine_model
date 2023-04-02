from flask import Flask, request, jsonify
import numpy as np
import keras
import tensorflow

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    data = np.array(list(data.values())).astype(float).reshape(1, 23)

    #load model
    model = keras.models.load_model("migraine_cls.h5")

    # make a prediction using the input values and the loaded model
    y_prediction = model.predict(data)
    #
    # # Make the prediction
    # prediction = y_prediction[0]

    # create dictionary to map numerical labels to string labels
    label_map = {
        0: 'Typical aura with migraine',
        1: 'Typical aura without migraine',
        2: 'Migraine without aura',
        3: 'Basilar-type aura',
        4: 'Sporadic hemiplegic migraine',
        5: 'Familial hemiplegic migraine',
        6: 'Other'
    }

    # convert predicted probabilities to labels using label_map
    y_pred = [label_map[pred] for pred in np.argmax(y_prediction, axis=1)]


    # Return the prediction as a JSON response
    return jsonify({'prediction': str(y_pred[0])})

if __name__ == '__main__':
    app.run(debug=True)


