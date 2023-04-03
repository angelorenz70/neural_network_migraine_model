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

    p1 = float(y_prediction[0][0]) * 100
    p2 = float(y_prediction[0][1]) * 100
    p3 = float(y_prediction[0][2]) * 100
    p4 = float(y_prediction[0][3]) * 100
    p5 = float(y_prediction[0][4]) * 100
    p6 = float(y_prediction[0][5]) * 100
    p7 = float(y_prediction[0][6]) * 100

    # Return the prediction as a JSON response
    return jsonify({'prediction': str(y_pred[0]),
                    'prob1': "{:.2f}".format(p1),
                    'prob2': "{:.2f}".format(p2),
                    'prob3': "{:.2f}".format(p3),
                    'prob4': "{:.2f}".format(p4),
                    'prob5': "{:.2f}".format(p5),
                    'prob6': "{:.2f}".format(p6),
                    'prob7': "{:.2f}".format(p7)
                    })

if __name__ == '__main__':
    app.run(debug=True)


