from flask import Flask, request, jsonify
import pickle

# Create app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('../model/model.pkl', 'rb'))

# Create API route
@app.route('/predict', methods=['POST'])
def predict():
    
    # Get data from request
    data = request.json
    
    # Extract values
    features = [
        data['N'],
        data['P'],
        data['K'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    ]
    
    # Predict
    prediction = model.predict([features])
    
    # Return result
    return jsonify({
        "recommended_crop": prediction[0]
    })

# Run server
if __name__ == '__main__':
    app.run(debug=True)