# Heart-Attack-Prediction-Webapp
An app that takes user medical data as input  and predicts if the patient  is likely to have  heart attack or not .

## How It Works

1. The user fills in a form with 11 medical details.
2. The Flask app sends those details to a trained Random Forest model (95% Acccuracy).
3. The app shows whether the person is likely to have heart disease or not.

## Inputs

| Input | Description |
|---|---|
| Age | Age in years |
| Gender | Male or female |
| Chest pain type | ATA, NAP, ASY or TA |
| Resting BP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| Fasting blood sugar | 1 if above 120 mg/dl, otherwise 0 |
| Resting ECG | Normal, ST or LVH |
| Max heart rate | Maximum heart rate achieved |
| Exercise angina | Whether exercise causes chest pain |
| Oldpeak | ST depression measured during exercise |
| ST slope | Slope of the peak exercise ST segment |

## Tech Stack

- **Flask** for the web server and routes
- **Bootstrap** for the input form
- **scikit-learn** for the pipeline (one-hot encoding and Random Forest classifier)
- **pandas** and **NumPy** for data handling
- **pickle** to save and load the trained model

