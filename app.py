import os, hmac, hashlib, time, json
from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import requests

load_dotenv()

# === Config ===
TV_WEBHOOK_TOKEN = os.getenv("TV_WEBHOOK_TOKEN", "")
API_KEY = os.getenv("THREECOMMAS_API_KEY", "")
API_SECRET = os.getenv("THREECOMMAS_API_SECRET", "")
BOT_ID = os.getenv("THREECOMMAS_BOT_ID", "")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# Init Flask
app = Flask(__name__)

# Init FinBERT (finance sentiment)
sent_clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def sentiment_score(text: str) -> float:
    """
    Returns +score for positive, -score for negative, 0 for neutral.
    """
    if not text or not text.strip():
        return 0.0
    res = sent_clf(text[:500])[0]  # guard length
    label = res["label"].lower()
    score = float(res["score"])
    if "positive" in label:
        return score
    if "negative" in label:
        return -score
    return 0.0

# --- 3Commas helpers ---
BASE_URL = "https://api.3commas.io/public/api"

def _sign(query: str) -> str:
    return hmac.new(API_SECRET.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

def _headers():
    return {
        "APIKEY": API_KEY,
        "Signature": "",  # set per-request after signing
        "Content-Type": "application/json"
    }

def start_deal(bot_id: str, pair: str, message: str = ""):
    """
    Starts a deal on a DCA bot by ID for the specified pair (e.g., 'BINANCE:BTCUSDT').
    Adjust the 'pair' formatting to your 3Commas exchange account if needed.
    """
    endpoint = "/ver1/bots/start_deal"
    # 3Commas expects query-string signing; to keep it simple, we send JSON body and also sign an empty query.
    query = ""  # no query params used; if you add, include them here like "bot_id=...&pair=..."
    url = BASE_URL + endpoint
    headers = _headers()
    headers["Signature"] = _sign(query)

    payload = {
        "bot_id": int(bot_id),
        "pair": pair,
        "message": message[:240]
    }

    if DRY_RUN:
        print(f"[DRY_RUN] Would call 3Commas start_deal: {payload}")
        return {"status": "dry_run"}

    r = requests.post(url, headers=headers, params={}, json=payload, timeout=20)
    try:
        data = r.json()
    except Exception:
        data = {"status_code": r.status_code, "text": r.text}
    print("3Commas response:", data)
    return data

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "invalid JSON"}), 400

    # Basic auth via shared token
    token = data.get("token", "")
    if TV_WEBHOOK_TOKEN and token != TV_WEBHOOK_TOKEN:
        return jsonify({"ok": False, "error": "bad token"}), 401

    # Expected TradingView fields (you control this in the alert message template)
    symbol = data.get("symbol")           # e.g., "BTCUSDT"
    exchange = data.get("exchange", "BINANCE").upper()
    side = data.get("side", "").lower()   # "buy" or "sell"
    note = data.get("note", "")           # any text (we run sentiment on this)
    confidence = float(data.get("confidence", 0))  # optional
    pair = f"{exchange}:{symbol}" if ":" not in symbol else symbol

    # Sentiment gate
    s = sentiment_score(note)
    # Simple decision: require alignment between side and sentiment
    # (buy if s>0.2, sell if s<-0.2). You can tune thresholds as you like.
    should_trade = False
    reason = ""
    if side == "buy" and s > 0.2:
        should_trade = True
        reason = f"BUY allowed (sentiment {s:.2f})"
    elif side == "sell" and s < -0.2:
        should_trade = True
        reason = f"SELL allowed (sentiment {s:.2f})"
    else:
        reason = f"Blocked by sentiment {s:.2f}"

    # Optional extra gate: confidence from TV (e.g., your strategy’s score)
    if confidence and confidence < 60:
        should_trade = False
        reason += f" | low TV confidence {confidence}%"

    print(f"[TV] pair={pair} side={side} sentiment={s:.2f} conf={confidence} note={note!r} -> {reason}")

    if not should_trade:
        return jsonify({"ok": True, "action": "skipped", "reason": reason})

    # Fire 3Commas
    if not (API_KEY and API_SECRET and BOT_ID):
        return jsonify({"ok": False, "error": "3Commas API not configured (API key/secret/bot id)"}), 500

    resp = start_deal(BOT_ID, pair, message=f"TV+FinBERT: {side} ({reason})")
    return jsonify({"ok": True, "action": "deal_started", "threecommas": resp})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
