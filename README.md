# 🌾 Crop Price Forecasting Dashboard

An interactive full-stack web application for forecasting market prices of crops like **Rice**, **Wheat**, and **Pepper** using multiple machine learning models — **ARIMA**, **Prophet**, **LSTM**, and **Random Forest**. The dashboard helps farmers, policymakers, and researchers visualize forecasts, compare model accuracy, and make informed decisions.

---

## 🚀 Features

- 📅 Predict future crop prices for a selected date
- 🧠 Multiple forecasting models: ARIMA, Prophet, LSTM, RF
- 📊 Comparative metrics: MAE, RMSE, R² for each model
- 🥇 Automatically highlights the best performing model
- 📈 Real vs. Predicted line chart visualizations
- 💻 Built with React (frontend) and Flask (backend)

---

## 🏗️ Project Structure

Market/
│
├── frontend/ # React UI (App.jsx, Components, CSS)
│
├── backend/
│ ├── src/
│ │ ├── api.py # Flask API endpoints
│ │ ├── train_all.py # Trains all models for a given crop
│ │ ├── evaluate.py # Evaluation metrics (MAE, RMSE, R²)
│ │ ├── preprocess.py # Preprocessing utils for ML models
│ │ ├── train_lstm.py # LSTM model training
│ │ ├── train_arima.py # ARIMA model training
│ │ ├── train_prophet.py # Prophet model training
│ │ ├── train_rf.py # Random Forest model training
│ │ ├── data_loader.py # Load historical crop data
│ │ └── generate_test_results.py # Save test results for dashboard
│
├── models/ # Pickled models, scalers, metrics
├── data/ # CSV files like rice_price.csv
└── requirements.txt # Python dependencies


---

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/CropForecastingDashboard.git
cd CropForecastingDashboard

 2. Backend Setup (Python + Flask)
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate    # macOS/Linux

pip install -r requirements.txt

python src/train_all.py --crop rice

python src/api.py
# API: http://localhost:5000

3. Frontend Setup (React)

cd frontend
npm install
npm run dev
# App: http://localhost:3000




