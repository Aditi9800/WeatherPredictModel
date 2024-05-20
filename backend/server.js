const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

let model;

// Load the trained model
try {
  const modelPath = path.join(__dirname, 'model.pkl');

  model = fs.readFileSync(modelPath);
  console.log('Model loaded successfully');
} catch (error) {
  console.error('Error loading model:', error);
  model = null;
}

app.post('/predict', (req, res) => {  
  const { Temperature_Celsius, Visibility_km, Wind_Speed_knots } = req.body;
  // console.log("predict1");
  // Validate input data
  if (Temperature_Celsius == null || Visibility_km == null || Wind_Speed_knots == null) {
    return res.status(400).json({ error: 'Invalid input' });
  }

  // Format input data for prediction
  const features = [Temperature_Celsius, Visibility_km, Wind_Speed_knots];
  // console.log("features "+features);
  // Use Python to load the model and make the prediction
  const options = {
    mode: 'text',
    pythonOptions: ['-u'],
    scriptPath: __dirname,
    args: features,
  };
  // console.log(options);

  PythonShell.run('predict.py', options).then(  results => {
    // console.log("predict6");
    // if (err) {
    //   console.error('Error:', err);
    //   return res.status(500).json({ error: 'Error making prediction' });
    // }
    // console.log("predict4");
    const prediction = results[0];
    res.json({ prediction: parseInt(prediction, 10) });
    // console.log("predict5");
  });
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
