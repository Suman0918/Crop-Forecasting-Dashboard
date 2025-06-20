import React from "react";

// ✅ Simulated (fake) model metrics per crop
const mockMetrics = {
  rice: {
    arima: { MAE: 4.8, RMSE: 5.6, R2: 0.42 },
    prophet: { MAE: 3.6, RMSE: 4.9, R2: 0.64 },
    lstm: { MAE: 2.1, RMSE: 3.1, R2: 0.81 },
    rf: { MAE: 2.7, RMSE: 3.6, R2: 0.73 },
  },
  wheat: {
    arima: { MAE: 6.5, RMSE: 7.8, R2: 0.39 },
    prophet: { MAE: 4.8, RMSE: 6.2, R2: 0.61 },
    lstm: { MAE: 3.3, RMSE: 4.4, R2: 0.75 },
    rf: { MAE: 4.0, RMSE: 5.1, R2: 0.69 },
  },
  pepper: {
    arima: { MAE: 32.0, RMSE: 40.5, R2: 0.28 },
    prophet: { MAE: 24.5, RMSE: 35.0, R2: 0.52 },
    lstm: { MAE: 16.2, RMSE: 22.7, R2: 0.72 },
    rf: { MAE: 19.4, RMSE: 26.1, R2: 0.65 },
  },
};

export default function MetricsTable({ results }) {
  // ✅ Extract crop name from incoming `results` object
  const crop = results?.crop || "rice";

  const metricsData = mockMetrics[crop] || {};

  return (
    <div style={{ marginTop: "1rem" }}>
      <h2
        style={{
          fontSize: "1.5rem",
          color: "red",
          marginBottom: "0.75rem",
        }}
      >
        Model Performance Metrics – {crop.toUpperCase()}
      </h2>
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>R²</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(metricsData).map(([model, metrics]) => {
            const r2 = Math.max(0, Math.min(1, metrics.R2)).toFixed(2);
            const mae = metrics.MAE.toFixed(2);
            const rmse = metrics.RMSE.toFixed(2);

            return (
              <tr key={model}>
                <td style={{ textTransform: "uppercase", fontWeight: "600" }}>
                  {model}
                </td>
                <td>{mae}</td>
                <td>{rmse}</td>
                <td
                  style={{
                    color:
                      r2 >= 0.75
                        ? "#16a34a" // green
                        : r2 >= 0.5
                        ? "#eab308" // yellow
                        : "#dc2626", // red
                    fontWeight: "bold",
                  }}
                >
                  {r2}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
