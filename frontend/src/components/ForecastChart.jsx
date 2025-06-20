import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import dayjs from "dayjs";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const COLORS = {
  arima: "rgba(239, 68, 68, 0.6)",
  prophet: "rgba(59, 130, 246, 0.6)",
  lstm: "rgba(16, 185, 129, 0.6)",
  rf: "rgba(168, 85, 247, 0.6)",
};

export default function ForecastChart({ results }) {
  const firstKey = Object.keys(results)[0];
  const rawDates = results[firstKey].test_dates;
  const labels = rawDates.map((d) => dayjs(d).format("MMM D, YYYY"));

  const datasets = Object.entries(results).map(([modelName, dataObj]) => {
    const color = COLORS[modelName] || "rgba(107, 114, 128, 0.6)";
    return {
      label: modelName.toUpperCase(),
      data: dataObj.predictions,
      borderColor: color,
      backgroundColor: color,
      tension: 0.3,
      pointRadius: 2,
      fill: false,
    };
  });

  const chartData = {
    labels,
    datasets,
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "bottom" },
      title: {
        display: true,
        text: "Model Predictions vs. Test Dates",
        font: { size: 18 },
      },
      tooltip: { mode: "index", intersect: false },
    },
    interaction: { mode: "nearest", axis: "x", intersect: false },
    scales: {
      x: {
        display: true,
        title: { display: true, text: "Date" },
        ticks: { maxRotation: 45, minRotation: 45 },
      },
      y: {
        display: true,
        title: { display: true, text: "Predicted Price" },
      },
    },
  };

  return (
    <div className="chart-container">
      <Line data={chartData} options={options} />
    </div>
  );
}
