# 🤖 BTC Intraday AI Trading Assistant - User Guide

## 🚨 IMPORTANT DISCLAIMER
This software provides algorithmic **market analysis**, NOT financial advice. 
Cryptocurrency trading involves high risk. **Past performance is not indicative of future results.** 
Always verify signals with your own analysis and never trade money you cannot afford to lose.

---

## 1. Quick Start

### ⚡ Installation
1.  **Install Requirements** (First time only):
    ```bash
    pip install -r requirements.txt
    ```
2.  **Initialize System** (Build Data & Models):
    ```bash
    python -m src.main_pipeline
    ```
    *Select 'y' (Yes) when asked to retrain.*

### 🚀 Daily Usage
To start the Live Assistant:
```bash
python -m src.live_trading_assistant
```
Follow the on-screen prompts to enter your capital (e.g., $1000). The assistant will monitor the market every 60 seconds.

---

## 2. How It Works (Simplified)

This AI system works like a professional analyst watching the charts 24/7.

1.  **Reads the Market**: It pulls live price data (Open, High, Low, Close, Volume) every 15 minutes.
2.  **Calculates Indicators**: It computes 20+ technical indicators like RSI, MACD, and Volatility.
3.  **Analyzes Patterns**: It uses a machine learning model (**XGBoost**) trained on years of Bitcoin history.
4.  **Makes a Probabilistic Guess**: It doesn't just say "Up" or "Down". It calculates the *probability* of a move.
    *   *Example: "There is a 72% chance price goes up by at least 0.5% in the next hour."*

---

## 3. Reading Signals

The assistant generates a "Trade Card" when it finds a high-confidence setup.

### Example Signal

```text
==================================================
🚀 RECOMMENDATION: BUY
==================================================
Confidence:     72%             <-- Higher = Better (Min 65% req)
Current Price:  $95,234.50
------------------------------
Position Size:  $500.00         <-- Recommended Trade Size
Stop Loss:      $94,850.00      <-- EXIT if price drops here
Take Profit:    $95,895.00      <-- EXIT if price rises here
------------------------------
Risk Amount:    $200.00         <-- Most you could lose (2%)
Pot. Profit:    $300.00         <-- Expected Gain (3%)
Risk:Reward:    1:1.50          <-- You win $1.50 for every $1 risked
==================================================
```

### Signal Types
*   **🚀 BUY**: The AI predicts price will rise. Enter a LONG position.
*   **🔻 SELL**: The AI predicts price will fall. Enter a SHORT position.
*   **✋ HOLD**: No clear direction. **Do not trade.**

---

## 4. Risk Management (Critical!)

The system is designed to protect your capital first, and grow it second.

### Position Sizing
The AI calculates position size automatically so that **you never risk more than 2% of your account on a single trade.**
*   *If Stop Loss is wide ($1,000 away)* -> Position size will be **Small**.
*   *If Stop Loss is tight ($200 away)* -> Position size will be **Larger**.

### Stop Loss (SL) & Take Profit (TP)
*   **SL**: Calculated based on volatility (2x ATR). **Always use a Stop Loss.**
*   **TP**: Set at 1.5x the risk. This ensures that even if you only win 45% of trades, you can still be profitable.

---

## 5. Best Practices

✅ **Confirm with Trend**: If Bitcoin is crashing (-5% day), be careful with BUY signals. Trend is your friend.
✅ **Check the News**: Don't trade during major events (Fed meetings, CPI data). The AI cannot read the news.
✅ **Verify Confidence**: Signals with 65-70% confidence are "Weak". Signals with >80% are "Strong".
⛔ **Don't Over-leverage**: The recommended position sizing assumes 1x leverage. If you use 10x leverage, cut the size by 10x!

---

## 6. Troubleshooting

**Issue**: "Model file not found."
*   **Solution**: Run `python -m src.main_pipeline` to train the model first.

**Issue**: "API Error / Timeout"
*   **Solution**: Check your internet connection. Binance API might be temporarily down. Wait 5 minutes.

**Issue**: "No trades for hours"
*   **Solution**: This is normal! The market is mostly noise. The AI waits for high-probability setups. Patience pays.

---

Happy Trading! 📉📈
