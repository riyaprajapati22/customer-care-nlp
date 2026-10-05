"""NLP engine: spelling -> intent -> sentiment -> retrieval (RAG) -> generation."""
import json, os, re, difflib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

BASE = os.path.dirname(__file__)
STOP = set("a an the is are was i my me to of for in on it and or do you we can how what when where with this that be at".split())
NEG = {"terrible","worst","awful","angry","useless","horrible","bad","disappointed","scam","fraud","pathetic","ridiculous","waste","rubbish","unacceptable","furious","annoyed","broken","damaged","wrong","late","never","not working"}
POS = {"thanks","thank","great","good","awesome","perfect","happy","love","helpful","excellent","nice"}
ESCALATE_INTENTS = {"human_request"}


ORDER_RE = re.compile(r"(?:ord[-_ ]?)?#?(\d{4,10})\b", re.I)
# Mock order database. Any other ID defaults to "in transit".
MOCK_ORDERS = {
    "10001": "has been delivered. If you cannot find it, check with neighbours or contact us within 48 hours.",
    "10002": "is being packed and will ship within 1 business day.",
    "10003": "is out for delivery and will reach you today.",
}
DEFAULT_ORDER = "is in transit and will arrive in a few days. You can follow live updates from My Orders."


def tokens(t): return re.findall(r"[a-z0-9]+", t.lower())


class CareAgent:
    def __init__(self):
        self.kb = json.load(open(os.path.join(BASE, "knowledge_base.json")))
        intents = json.load(open(os.path.join(BASE, "intents.json")))
        X = [s for v in intents.values() for s in v]
        y = [k for k, v in intents.items() for _ in v]
        self.intent_clf = make_pipeline(
            make_union(TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            LogisticRegression(max_iter=1000, C=10)).fit(X, y)
        docs = [d["topic"] + ". " + d["text"] for d in self.kb]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.M = self.vec.fit_transform(docs)
        self.vocab = {w for d in docs for w in tokens(d)} | {w for s in X for w in tokens(s)}
        self.llm = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic; self.llm = anthropic.Anthropic()
            except Exception: pass

    # 1. spelling correction against domain vocabulary
    def correct(self, text):
        out = []
        for w in tokens(text):
            if w in self.vocab or w.isdigit() or len(w) < 4: out.append(w); continue
            m = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.8)
            out.append(m[0] if m else w)
        return " ".join(out)

    # 2. intent
    def intent(self, text):
        p = self.intent_clf.predict_proba([text])[0]
        i = int(np.argmax(p)); return self.intent_clf.classes_[i], float(p[i])

    # 3. sentiment (lexicon-based, returns -1..1)
    def sentiment(self, raw):
        t = raw.lower(); w = set(tokens(t))
        neg = len(w & NEG) + sum(1 for p in ("not working", "very bad", "so bad") if p in t) + t.count("!") // 2
        pos = len(w & POS)
        if raw.isupper() and len(raw) > 6: neg += 1
        return max(-1.0, min(1.0, (pos - neg) / 3))

    # 4. retrieval
    def retrieve(self, query, k=2):
        s = (self.M @ self.vec.transform([query]).T).toarray().ravel()
        idx = np.argsort(-s)[:k]
        return [(self.kb[i], float(s[i])) for i in idx if s[i] > 0]

    # 5. generation
    def generate(self, user, passages, history, intent, sent):
        ctx = "\n".join(f"- {p['topic']}: {p['text']}" for p, _ in passages)
        if self.llm:
            try:
                sys = ("You are a friendly customer care agent. Answer ONLY using the knowledge below. "
                       "Be concise (max 4 sentences), empathetic if the customer is upset, and ask for the order ID if needed. "
                       "If the knowledge does not cover it, say you will connect them to a human.\nKNOWLEDGE:\n" + ctx)
                msgs = [{"role": r, "content": c} for r, c in history[-6:]] + [{"role": "user", "content": user}]
                r = self.llm.messages.create(model="claude-sonnet-4-6", max_tokens=300, system=sys, messages=msgs)
                return r.content[0].text, "llm"
            except Exception: pass
        # offline generative fallback: compose reply from retrieved passages
        opener = ("I'm really sorry about the trouble. " if sent < -0.2 else
                  "Happy to help! " if intent in ("product_info", "payment", "account") else "")
        body = passages[0][0]["text"]
        extra = f" Also, {passages[1][0]['text'][0].lower() + passages[1][0]['text'][1:]}" if len(passages) > 1 and passages[1][1] > 0.6 * passages[0][1] and passages[1][1] > 0.15 else ""
        return opener + body + extra + " Is there anything else I can help with?", "rag"

    def reply(self, message, history):
        clean = self.correct(message)
        intent, conf = self.intent(clean)
        sent = self.sentiment(message)
        meta = {"intent": intent, "confidence": round(conf, 2), "sentiment": round(sent, 2), "corrected": clean}
        if intent == "greeting" and len(tokens(message)) <= 4:
            return "Hello! I'm your customer care assistant. Ask me about orders, returns, payments or anything else.", {**meta, "mode": "rule"}
        if intent == "thanks" and len(tokens(message)) <= 6:
            return "You're welcome! Have a great day.", {**meta, "mode": "rule"}
        if intent in ESCALATE_INTENTS:
            return "Connecting you to a human agent. Support hours are Mon-Sat, 9 AM to 8 PM IST (1800-000-1234).", {**meta, "mode": "escalate"}
        # order status flow: ask for ID, remember it, look it up
        m = ORDER_RE.search(message)
        oid = m.group(1) if m else None
        asked = bool(history) and history[-1][0] == "assistant" and "order ID" in history[-1][1]
        how_q = "how" in tokens(message) and "where" not in tokens(message)
        if (intent == "order_status" and not how_q) or (asked and oid) or (oid and len(tokens(message)) <= 2):
            if not oid:
                for r, c in reversed(history):
                    mm = ORDER_RE.search(c) if r == "user" else None
                    if mm: oid = mm.group(1); break
            if oid:
                return f"Your order #{oid} " + MOCK_ORDERS.get(oid, DEFAULT_ORDER), {**meta, "mode": "order_lookup"}
            return "Sure, I can check that. Please share your order ID (for example ORD12345).", {**meta, "mode": "rule"}
        # follow-up handling: short query -> enrich with previous user turn
        passages = self.retrieve(clean)
        users = [c for r, c in history if r == "user"]
        if users and (not passages or passages[0][1] < 0.25) and len(tokens(message)) <= 5:
            passages = self.retrieve(self.correct(users[-1]) + " " + clean)
        if sent <= -0.67 or not passages or passages[0][1] < 0.12:
            msg = ("I'm sorry about this experience. I'm escalating you to a human agent who can resolve it quickly. "
                   if sent <= -0.67 else "I'm not sure I can answer that accurately, so let me connect you to a human agent. ")
            if passages and sent <= -0.67: msg += passages[0][0]["text"]
            return msg, {**meta, "mode": "escalate"}
        text, mode = self.generate(message, passages, history, intent, sent)
        return text, {**meta, "mode": mode, "sources": [p["topic"] for p, _ in passages]}
