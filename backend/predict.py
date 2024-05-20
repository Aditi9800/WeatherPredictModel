import sys
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



# Get input features from the command line arguments
features = np.array([float(x) for x in sys.argv[1:]]).reshape(1, -1)

# Make prediction
prediction = model.predict(features)[0]


# Print prediction (this will be captured by the Node.js server)
print(int(prediction))
