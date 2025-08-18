 
from flask import Flask, render_template_string, send_from_directory
import plotly.graph_objs as go
import pandas as pd
import os
import json

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.debt_model import DivineSupremeDebtPredictor

app = Flask(__name__)

LAYOUT = """
<!doctype html>
<html>
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>DIVINE Debt Dashboard</title>
		<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
		<style>body{font-family: Arial, sans-serif; margin: 20px;}</style>
	</head>
	<body>
		<h2>DIVINE Debt Dashboard</h2>
		<div id="plot"></div>
		<div id="meta">{{ meta }}</div>
		<script>
			const fig = {{ fig_json | safe }};
			Plotly.newPlot('plot', fig.data, fig.layout || {});
		</script>
	</body>
</html>
"""


def build_dashboard():
		dp = DivineSupremeDebtPredictor()
		dp.load_debt_data('data/raw/')
		dp.prepare_debt_time_series()
		dp.create_divine_debt_features()
		# Train a light model for dashboard responsiveness
		try:
				from sklearn.ensemble import RandomForestRegressor
				dp.models = {'Smoke_RF': RandomForestRegressor(n_estimators=20, random_state=42)}
		except Exception:
				pass
		dp.train_divine_models()
		preds = dp.generate_debt_predictions() or {}

		ts = dp.debt_ts.copy()
		ts = ts[[dp.target_col]].dropna()

		trace_hist = go.Scatter(x=ts.index, y=ts[dp.target_col], mode='lines+markers', name='Total Debt')

		# Add ensemble point if available
		fig = {'data': [trace_hist], 'layout': {'title': 'Total Public Debt (Historical)'}}
		meta = {
				'last_date': str(ts.index.max()) if not ts.empty else None,
				'last_value': float(ts[dp.target_col].iloc[-1]) if not ts.empty else None,
				'predictions': {k: float(v) for k, v in preds.items()}
		}

		# Add prediction marker
		if 'Divine_Ensemble' in preds and ts.index.max() is not None:
				next_date = pd.to_datetime(ts.index.max()) + pd.offsets.MonthBegin(1)
				pred_trace = go.Scatter(x=[next_date], y=[preds['Divine_Ensemble']], mode='markers', marker={'size':12,'color':'red'}, name='Ensemble Prediction')
				fig['data'].append(pred_trace)

		return fig, json.dumps(meta, default=str)


@app.route('/')
def index():
		fig, meta = build_dashboard()
		return render_template_string(LAYOUT, fig_json=json.dumps(fig, default=str), meta=meta)


if __name__ == '__main__':
		app.run(debug=True, port=8050)

