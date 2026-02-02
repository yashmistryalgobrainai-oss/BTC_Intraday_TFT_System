import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import threading
import time
from datetime import datetime
import os
import sys

# Ensure src path is available
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.live_trading_assistant import LiveTradingAssistant
from src.config import Config

# Initialize Assistant
model_path = 'trading_assistant_v2.json'
bot = LiveTradingAssistant(model_path)

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], meta_tags=[
    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
])
app.title = "BTC AI Trading Dashboard"

# --- GLOBAL STATE (Thread-safe ideally, simple for dashboard) ---
latest_data = {
    'price': 0,
    'signal': 'WAITING',
    'confidence': 0,
    'atr': 0,
    'timestamp': datetime.now()
}

def update_market_data():
    """Background thread to keep data fresh."""
    global latest_data
    while True:
        try:
            df = bot.fetch_live_candles(limit=150)
            if df is not None:
                # Process features
                try:
                    df_processed = bot.feature_engineer.fit_transform(df)
                    latest_idx = -2
                    row = df_processed.iloc[latest_idx:latest_idx+1]
                    price = df['close'].iloc[-1]
                    atr = df_processed['atr'].iloc[latest_idx]
                    
                    # Predict
                    analysis = bot.model.predict_signal(row)
                    
                    latest_data = {
                        'price': price,
                        'signal': analysis['signal'],
                        'confidence': analysis['confidence'],
                        'atr': atr,
                        'timestamp': datetime.now(),
                        'history': df.tail(50) # Save history for chart
                    }
                except Exception as e:
                    print(f"Analysis Error: {e}")
        except Exception as e:
            print(f"Fetch Error: {e}")
        
        time.sleep(60) # Update every minute

# Start Background Thread
t = threading.Thread(target=update_market_data)
t.daemon = True
t.start()

# --- LAYOUT ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H2("🤖 BTC/USDT AI TRADING SYSTEM", className="text-center text-primary mb-4"), width=12)
    ], className="mt-4"),
    
    # KPIs
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Current Price", className="card-title"),
                html.H2(id="live-price", className="text-white"),
            ])
        ], color="dark", inverse=True), width=12, md=4),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Recommendation", className="card-title"),
                html.H2(id="recommendation", className="text-warning"),
            ])
        ], color="dark", inverse=True), width=12, md=4),
        
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("AI Confidence", className="card-title"),
                dcc.Graph(id="confidence-gauge", style={'height': '150px'}),
            ])
        ], color="dark", inverse=True), width=12, md=4),
    ], className="mb-4"),
    
    # Calculator & Chart
    dbc.Row([
        # Position Calculator
        dbc.Col(dbc.Card([
            dbc.CardHeader("💰 Position Size Calculator"),
            dbc.CardBody([
                dbc.Label("Trading Capital ($)"),
                dbc.Input(id="capital-input", type="number", value=1000, step=100),
                html.Br(),
                html.Div(id="calc-output")
            ])
        ], color="secondary", inverse=True), width=12, md=4),
        
        # Chart
        dbc.Col(dbc.Card([
            dbc.CardHeader("📈 Live Market Chart"),
            dbc.CardBody([
                dcc.Graph(id="price-chart")
            ])
        ], color="secondary", inverse=True), width=12, md=8),
    ]),
    
    # Footer / Status
    dbc.Row([
        dbc.Col(html.P(id="last-update", className="text-muted text-center mt-4"), width=12)
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=10*1000, # Update UI every 10 seconds (Data updates every 60s)
        n_intervals=0
    )
], fluid=True)

# --- CALLBACKS ---

@app.callback(
    [Output("live-price", "children"),
     Output("recommendation", "children"),
     Output("confidence-gauge", "figure"),
     Output("price-chart", "figure"),
     Output("last-update", "children"),
     Output("calc-output", "children")],
    [Input("interval-component", "n_intervals"),
     Input("capital-input", "value")]
)
def update_dashboard(n, capital):
    if not capital: capital = 0
    
    price = latest_data['price']
    conf = latest_data['confidence']
    sig = latest_data['signal']
    atr = latest_data['atr']
    ts = latest_data['timestamp']
    
    # 1. Recommendation Text & Color
    if conf < 0.65 or sig == 0:
        rec_text = "✋ HOLD"
        rec_color = "text-warning"
    elif sig == 1:
        rec_text = "🚀 BUY"
        rec_color = "text-success"
    else:
        rec_text = "🔻 SELL"
        rec_color = "text-danger"
    
    rec_div = html.Span(rec_text, className=rec_color)
    
    # 2. Confidence Gauge
    gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = conf * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence %"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00cc96"},
            'steps': [
                {'range': [0, 50], 'color': "#ff5555"},
                {'range': [50, 65], 'color': "#ffcc00"},
                {'range': [65, 100], 'color': "#00cc96"}],
        }
    ))
    gauge.update_layout(margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})

    # 3. Chart
    chart = go.Figure()
    if 'history' in latest_data:
        df = latest_data['history']
        chart.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name='BTC/USDT'
        ))
        chart.update_layout(
            template="plotly_dark",
            margin=dict(l=30, r=30, t=30, b=30),
            height=400,
            xaxis_rangeslider_visible=False
        )

    # 4. Calculator Output
    calc_res = []
    if sig != 0 and conf >= 0.65:
        # Calculate params using bot logic
        params = bot._calculate_trade_params(1 if sig==1 else -1, price, atr, float(capital))
        if params:
            calc_res = [
                html.H5("Recommended Setup:", className="mt-2 text-info"),
                html.P(f"Position Value: ${params['value']:,.2f}", className="mb-1"),
                html.P(f"Stop Loss: ${params['sl']:,.2f}", className="text-danger mb-1"),
                html.P(f"Take Profit: ${params['tp']:,.2f}", className="text-success mb-1"),
                html.Small(f"Max Risk: ${params['risk']:.2f} (2%)")
            ]
    else:
        calc_res = html.P("Waiting for valid signal...", className="text-muted mt-3")

    last_update_text = f"Last Data Update: {ts.strftime('%H:%M:%S')}"
    
    return f"${price:,.2f}", rec_div, gauge, chart, last_update_text, calc_res

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
