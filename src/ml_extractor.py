"""TF-IDF + Logistic Regression, trained on weakly aligned SROIE box text."""
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from src.config import ROOT, FIELDS
from src.normalize import find_amounts_in_line, find_dates

MODEL_PATH = ROOT / "models" / "receipt_model.joblib"

def canonical(s):
    return re.sub(r"[^A-Z0-9]+", " ", unicodedata.normalize("NFKC", s or "").upper()).strip()

def line_features(lines):
    clean = [canonical(x) for x in lines]
    n = max(1, len(clean)-1)
    return [f"POS_{min(9,int(10*i/n))} {x} PREV {clean[i-1][-60:] if i else 'START'} NEXT {clean[i+1][:60] if i+1<len(clean) else 'END'}" for i,x in enumerate(clean)]

def score(line, target):
    a,b = canonical(line),canonical(target)
    if not a or not b: return 0.0
    if a in b or b in a: return len(a if len(a)<len(b) else b)/max(len(a),len(b))
    return max(0.65*len(set(a.split())&set(b.split()))/max(1,len(set(a.split()))),0.65*SequenceMatcher(None,a,b).ratio())

def label_lines(lines,entities):
    labels=["OTHER"]*len(lines)
    for field in ("date","total","company","address"):
        target=entities.get(field)
        if not target: continue
        pairs=[(score(line,target),i) for i,line in enumerate(lines)]
        if field=="address":
            for value,i in pairs:
                if value>=0.38 and len(canonical(lines[i]))>=5 and labels[i]=="OTHER": labels[i]="ADDRESS"
        elif pairs:
            value,i=max(pairs)
            if value >= (0.52 if field in ("date","total") else 0.45) and labels[i]=="OTHER":
                labels[i]=field.upper()
    return labels

def build_estimator():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline,FeatureUnion
    return Pipeline([
        ("tfidf",FeatureUnion([
            ("word",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
            ("char",TfidfVectorizer(analyzer="char",ngram_range=(2,5),sublinear_tf=True))
        ])),
        ("classifier",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))
    ])

def predict_fields(lines,model):
    out={f:None for f in FIELDS}
    if not lines: return out
    x=line_features(lines)
    predicted=model.predict(x)
    probabilities=model.predict_proba(x)
    classes=list(model.classes_)
    for field in ("company","date","total"):
        label=field.upper()
        if label not in classes: continue
        options=[i for i,pred in enumerate(predicted) if pred==label]
        if not options: continue
        i=max(options,key=lambda j:probabilities[j,classes.index(label)])
        raw=lines[i].strip()
        if field=="date":
            matches=find_dates(raw)
            out[field]=matches[0][0] if matches else None
        elif field=="total":
            matches=find_amounts_in_line(raw)
            out[field]=f"{matches[-1][1]:.2f}" if matches else None
        else: out[field]=raw or None
    addresses=[lines[i].strip() for i,p in enumerate(predicted) if p=="ADDRESS"]
    out["address"]=" ".join(addresses) if addresses else None
    return out

def load_model(path=MODEL_PATH):
    import joblib
    if not Path(path).is_file():
        raise FileNotFoundError(f"Chưa có mô hình {path}; hãy chạy python scripts/train_model.py")
    # joblib is executable: only load locally created, trusted models.
    return joblib.load(path)
