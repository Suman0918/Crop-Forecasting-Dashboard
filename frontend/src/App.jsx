import React, { useState, useEffect } from "react";
import ForecastForm from "./components/ForecastForm";
import ForecastChart from "./components/ForecastChart";
import MetricsTable from "./components/MetricsTable";
import FutureResults from "./components/FutureResults";

export default function App() {
  const [historicalResults, setHistoricalResults] = useState(null);
  const [futureResults, setFutureResults] = useState(null);
  const [selectedCrop, setSelectedCrop] = useState("rice");
  const [darkMode, setDarkMode] = useState(false);

  // Toggle class on <body> when theme changes
  useEffect(() => {
    document.body.className = darkMode ? "dark-mode" : "light-mode";
  }, [darkMode]);

  return (
    <div className="app-container">
      {/* 🌙 Theme Toggle */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1
          style={{
            marginBottom: "1.5rem",
            color: darkMode ? "#10b981" : "#047857",
            fontSize: "1.75rem",
          }}
        >
          Crop Price Forecasting Dashboard
        </h1>
        <button
          onClick={() => setDarkMode(!darkMode)}
          style={{
            padding: "0.5rem 1rem",
            background: darkMode ? "#facc15" : "#1e293b",
            color: darkMode ? "#1e293b" : "#f8fafc",
            borderRadius: "8px",
            border: "none",
            fontWeight: "bold",
            cursor: "pointer",
            transition: "all 0.3s ease",
          }}
        >
          {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
        </button>
      </div>

      <div className="card">
        <ForecastForm
          onCropChange={(crop) => setSelectedCrop(crop)}
          onHistoricalData={(data) => {
            setHistoricalResults(data);
            setFutureResults(null);
          }}
          onFutureData={(data) => setFutureResults(data)}
        />

        {historicalResults && (
          <>
            <ForecastChart results={historicalResults} />

            {/* Side-by-side layout */}
            <div
              style={{
                display: "flex",
                gap: "1.5rem",
                flexWrap: "wrap",
                marginTop: "1rem",
              }}
            >
              {/* Metrics Table */}
              <div
                style={{
                  flex: 1,
                  minWidth: "300px",
                  padding: "1rem",
                  borderRadius: "8px",
                }}
              >
                <MetricsTable results={{ crop: selectedCrop }} />
              </div>

              {/* Future Forecasts */}
              <div
                style={{
                  flex: 1,
                  minWidth: "300px",
                  padding: "1rem",
                  borderRadius: "8px",
                  paddingBlock: "1.8rem",
                }}
              >
                {futureResults && <FutureResults data={futureResults} />}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
