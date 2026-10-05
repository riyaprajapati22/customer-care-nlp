# Intelligent Customer Care Agent (NLP Mini Project)

Free-form, context-aware support chatbot. Answers are generated from a knowledge base (RAG), not matched to fixed Q&A.

## Pipeline
spelling correction -> intent classification (TF-IDF + Logistic Regression) -> sentiment (lexicon)
-> retrieval (TF-IDF cosine over knowledge_base.json) -> generation -> escalation to human.
Follow-ups ("what about refunds?") reuse the previous user turn for retrieval.

## Run
    pip install -r requirements.txt
    python app.py            # open http://127.0.0.1:5000

Optional: `export ANTHROPIC_API_KEY=...` for LLM-written answers grounded in the retrieved passages.
Without a key, an offline generator composes replies from the retrieved passages.

## Extend
Add entries to `knowledge_base.json` and example phrases to `intents.json`; no code changes needed.
# customer-care-nlp
