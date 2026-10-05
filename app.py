from flask import Flask, request, jsonify, render_template, session
import secrets
from nlp_engine import CareAgent

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
agent = CareAgent()
HISTORY = {}  # session id -> [(role, text)]


@app.route("/")
def index():
    session.setdefault("sid", secrets.token_hex(8))
    return render_template("index.html")


@app.post("/chat")
def chat():
    sid = session.setdefault("sid", secrets.token_hex(8))
    msg = (request.json or {}).get("message", "").strip()
    if not msg: return jsonify(error="Type a message first."), 400
    hist = HISTORY.setdefault(sid, [])
    text, meta = agent.reply(msg, hist)
    hist += [("user", msg), ("assistant", text)]
    return jsonify(reply=text, meta=meta)


@app.post("/reset")
def reset():
    HISTORY.pop(session.get("sid"), None); return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True)
