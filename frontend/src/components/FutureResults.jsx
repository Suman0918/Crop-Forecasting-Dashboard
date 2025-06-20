import React from "react";

export default function FutureResults({ data }) {
  const { date, predictions, best_model, best_prediction } = data;
  const modelKeys = Object.keys(predictions);

  return (
    <div className="future-results">
      <h3 style={{ fontSize: "1.25rem", color: "red" }}>
        Future Forecasts for {date}
      </h3>
      <ul>
        {modelKeys.map((m) => {
          const predValue = predictions[m].toFixed(2);
          const isBest = m === best_model;
          return (
            <li key={m} className={isBest ? "future-best" : ""}>
              <strong>{m.toUpperCase()}:</strong> ${predValue}{" "}
              {isBest && "(Best Model)"}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
