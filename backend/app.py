import os, json, time, re, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.sparse import hstack

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
ART  = os.path.join(BASE, 'artifacts')

def load_pkl(name):
    with open(os.path.join(ART, name), 'rb') as f:
        return pickle.load(f)

def load_json(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)

print("Loading models...")
svm        = load_pkl('svm_hybrid.pkl')
tfidf      = load_pkl('tfidf.pkl')
meta_pre   = load_pkl('meta_preprocessor.pkl')
lr_terms   = load_json('lr_terms.json')
s2_cache   = load_json('stage2_cache.json')
meta_info  = load_json('top_metadata.json')
M          = load_json('model_meta.json')
print(f"✅ Models loaded — SVM {M['svm_tps']:,} preds/sec | threshold [{M['low_thresh']},{M['high_thresh']}]")

META_BINARY = ['telecommuting','has_company_logo','has_questions']
META_CAT    = ['employment_type','required_experience','required_education','industry','function']
ALL_META_CAT = META_CAT + ['location_clean']
TEXT_COLS   = ['title','company_profile','description','requirements','benefits']

def clean(s):
    if not s or str(s).strip() == '': return ''
    s = str(s).lower()
    s = re.sub(r'http\S+|www\.\S+', 'url', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def build_features(fields):
    text = ' '.join(clean(fields.get(c,'')) for c in TEXT_COLS)

    row = {}
    for c in META_BINARY:
        v = fields.get(c, 0)
        row[c] = int(v) if v not in ('', None) else 0
    for c in META_CAT:
        row[c] = str(fields.get(c,'Unknown')) or 'Unknown'
    loc = fields.get('location','Unknown') or 'Unknown'
    row['location_clean'] = loc.split(',')[0].strip() if ',' in loc else loc

    df_row = pd.DataFrame([row])
    X_text = tfidf.transform([text])
    X_meta = meta_pre.transform(df_row)
    X_hyb  = hstack([X_text, X_meta])
    return text, X_text, X_hyb

def get_fraud_terms(X_text_vec, top_n=8):
    fn    = tfidf.get_feature_names_out()
    coefs = {t['term']: t['coef'] for t in lr_terms['fraud']}
    cx    = X_text_vec.tocoo()
    hits  = []
    for idx, val in zip(cx.col, cx.data):
        term = fn[idx]
        if term in coefs:
            hits.append({"term": term, "score": round(float(coefs[term] * val), 3)})
    hits.sort(key=lambda x: x['score'], reverse=True)
    return hits[:top_n]

def find_s2(s1_prob):
    return min(s2_cache, key=lambda x: abs(x['s1_prob'] - s1_prob))

@app.route('/')
def serve_frontend():
    return send_from_directory(os.path.join(BASE, '../frontend'), 'index.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok", "svm_tps": M['svm_tps'], "loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    fields = request.get_json(force=True).get('fields', {})
    if not any(fields.get(c,'').strip() for c in TEXT_COLS):
        return jsonify({"error": "No text provided"}), 400

    t0 = time.perf_counter()
    text, X_text_vec, X_hyb = build_features(fields)
    s1_prob = float(svm.predict_proba(X_hyb)[0, 1])
    s1_ms   = round((time.perf_counter() - t0) * 1000, 2)

    zone = ('LOW_RISK'  if s1_prob < M['low_thresh']  else
            'HIGH_RISK' if s1_prob >= M['high_thresh'] else 'REVIEW')

    fraud_terms = get_fraud_terms(X_text_vec)

    resp = {
        "stage1": {
            "prob":    round(s1_prob, 4),
            "time_ms": s1_ms,
            "zone":    zone,
            "gpu_used": False,
        },
        "stage2": None,
        "fraud_terms": fraud_terms,
        "thresholds":  {"low": M['low_thresh'], "high": M['high_thresh']},
        "stats": {
            "svm_tps":         M['svm_tps'],
            "bert_tps":        M['bert_tps'],
            "escalation_rate": M['escalation_rate'],
            "reduction_factor":M['reduction_factor'],
            "cost_bert_1m":    M['cost_bert_per_1m'],
            "cost_ts_1m":      M['cost_ts_per_1m'],
            "gpu_hrs_bert":    M['gpu_hrs_bert_annual'],
            "gpu_hrs_ts":      M['gpu_hrs_ts_annual'],
            "low_pct":         M['low_risk_pct'],
            "review_pct":      M['review_pct'],
            "high_pct":        M['high_risk_pct'],
            "svm_f1":          M['svm_f1'],
            "svm_roc":         M['svm_roc_auc'],
            "svm_brier":       M['svm_brier'],
            "svm_fp":          M['svm_fp'],
            "svm_fn":          M['svm_fn'],
        }
    }

    if zone == 'REVIEW':
        t2    = time.perf_counter()
        match = find_s2(s1_prob)
        s2_ms = round((time.perf_counter() - t2) * 1000, 2)
        resp["stage2"] = {
            "prob":        match['s2_prob'],
            "text_weight": match['text_weight'],
            "meta_weight": match['meta_weight'],
            "time_ms":     s2_ms,
            "gpu_used":    True,
            "note":        "Pre-computed on 30 representative REVIEW-zone samples from test set",
            "closest_s1":  match['s1_prob'],
            "top_terms":   match.get('top_terms', []),
        }

    return jsonify(resp)

@app.route('/stats')
def stats():
    return jsonify(M)

@app.route('/terms')
def terms():
    return jsonify({"fraud": lr_terms['fraud'][:20], "legit": lr_terms['legit'][:20],
                    "metadata": meta_info})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
