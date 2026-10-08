# Mobile Price Prediction Dashboard

This dashboard uses the **exact preprocessing and model pipeline from `Mobile_Price_Final_Model.ipynb`**.

## Final model
Gradient Boosting Regressor

- n_estimators = 500
- learning_rate = 0.05
- max_depth = 3
- random_state = 42

## First run
Open this folder in VS Code and run:

```powershell
pip install -r requirements.txt
python app.py
```

The application automatically runs `train_model.py` if the model artifacts do not exist. It then starts the Flask dashboard.

Open:

```text
http://127.0.0.1:5000
```

## If automatic training cannot download the dataset
Download/place `smartphones.csv` inside the `data` folder and run:

```powershell
python train_model.py
python app.py
```

The dataset URL used by the notebook is the same URL used by `train_model.py`.

## Prediction pipeline
The dashboard uses the same 26 features, feature engineering, categorical encoding, numerical scaling, train/test split, and Gradient Boosting configuration as the notebook. The five engineered features are calculated automatically during prediction:

- total_camera_count
- screen_pixels
- ppi
- charging_efficiency
- premium_features

No LLM or generative-AI functionality is used.
