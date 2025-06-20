import React, { useState } from "react";
import axios from "axios";

const API_BASE = "http://localhost:5000";

export default function ForecastForm({ onHistoricalData, onFutureData, onCropChange }) {
  const [crop, setCrop] = useState("rice");
  const [model, setModel] = useState("all");
  const [date, setDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (!date) {
      setError("Please select a future date.");
      return;
    }
    setLoading(true);
    try {
      // 1) Fetch historical results
      const histResp = await axios.get(`${API_BASE}/predict`, {
        params: { crop, model },
      });
      onHistoricalData(histResp.data.results);

      // 2) Fetch future single-date forecast
      const futResp = await axios.get(`${API_BASE}/predict_future`, {
        params: { crop, model, date },
      });
      onFutureData(futResp.data);
    } catch (err) {
      console.error(err);
      setError("An error occurred while fetching data. Check your backend.");
    }
    setLoading(false);
  };

  const handleCropChange = (e) => {
    const newCrop = e.target.value;
    setCrop(newCrop);
    if (onCropChange) {
      onCropChange(newCrop); // ✅ Notify parent (App.jsx)
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="crop-select">Crop</label>
        <select
          id="crop-select"
          value={crop}
          onChange={handleCropChange} // ✅ Use handler
        >
          <option value="rice">Rice</option>
          <option value="wheat">Wheat</option>
          <option value="pepper">Pepper</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="all">All Models</option>
          <option value="arima">ARIMA</option>
          <option value="prophet">Prophet</option>
          <option value="lstm">LSTM</option>
          <option value="rf">RF</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="date-input">Future Date</label>
        <input
          id="date-input"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
        />
      </div>

      {error && <p style={{ color: "red", marginBottom: "0.75rem" }}>{error}</p>}

      <button type="submit" disabled={loading}>
        {loading ? "Fetching…" : "Fetch Forecast"}
      </button>
    </form>
  );
}
