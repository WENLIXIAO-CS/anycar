/* charts.js — Chart.js live charts for MPPI planner visualization */

Chart.defaults.color = '#aaa';

const ACTION_LABELS = {
  holonomic: ['v_fwd', 'v_side', 'omega'],
  robocasa_holonomic: ['v_fwd', 'v_side', 'omega'],
  unicycle: ['accel', 'steer'],
  diff_drive: ['v_left', 'v_right'],
};

const ACTION_COLORS = ['#f59e0b', '#10b981', '#8b5cf6', '#ef4444', '#06b6d4'];

let distChart = null;
let actionChart = null;

window.initCharts = function () {
  const distCtx = document.getElementById('distChart').getContext('2d');
  distChart = new Chart(distCtx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Position Error (m)',
          data: [],
          borderColor: '#38bdf8',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.1,
          fill: false,
        },
        {
          label: 'Tolerance',
          data: [],
          borderColor: '#22c55e',
          borderWidth: 1,
          borderDash: [5, 3],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          display: true,
          title: { display: false },
          ticks: { maxTicksLimit: 6, color: '#aaa' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          display: true,
          beginAtZero: true,
          title: { display: false },
          ticks: { color: '#aaa' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });

  const actCtx = document.getElementById('actionChart').getContext('2d');
  actionChart = new Chart(actCtx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { maxTicksLimit: 6, color: '#aaa' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          ticks: { color: '#aaa' },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { boxWidth: 12, font: { size: 10 }, color: '#ccc' },
        },
      },
    },
  });
};

window.resetCharts = function (actionLabels) {
  distChart.data.labels = [];
  distChart.data.datasets[0].data = [];
  distChart.data.datasets[1].data = [];
  distChart.update();

  actionChart.data.labels = [];
  actionChart.data.datasets = actionLabels.map(function (label, i) {
    return {
      label: label,
      data: [],
      borderColor: ACTION_COLORS[i % ACTION_COLORS.length],
      borderWidth: 1.5,
      pointRadius: 0,
      tension: 0.1,
      fill: false,
    };
  });
  actionChart.update();
};

window.updateCharts = function (stepData) {
  var step = stepData.step;

  distChart.data.labels.push(step);
  distChart.data.datasets[0].data.push(stepData.pos_err);
  distChart.data.datasets[1].data.push(0.05);
  distChart.update();

  actionChart.data.labels.push(step);
  stepData.action.forEach(function (val, i) {
    if (actionChart.data.datasets[i]) {
      actionChart.data.datasets[i].data.push(val);
    }
  });
  actionChart.update();
};
