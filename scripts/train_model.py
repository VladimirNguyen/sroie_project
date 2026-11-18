"""Train on SROIE box transcription; evaluate validation and test separately."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
from sklearn.metrics import accuracy_score,classification_report,f1_score
from src.config import ROOT
from src.data_loader import get_splits
from src.evaluate import make_record,compute_metrics
from src.ml_extractor import MODEL_PATH,build_estimator,label_lines,line_features,predict_fields

def flatten(samples):
    x,y,n=[],[],0
    for s in samples:
        if not s.box_path or not s.entity_path: continue
        lines,entities=s.gt_lines(),s.labels()
        if not lines or not entities: continue
        x.extend(line_features(lines))
        y.extend(label_lines(lines,entities))
        n+=1
    return x,y,n

def evaluate(model,samples):
    x,y,n=flatten(samples)
    d={"receipts":n,"lines":len(y)}
    if y:
        yp=model.predict(x)
        d["line_accuracy_weak"]=float(accuracy_score(y,yp))
        d["line_macro_f1_weak"]=float(f1_score(y,yp,average="macro",zero_division=0))
        d["line_report"]=classification_report(y,yp,zero_division=0,output_dict=True)
    records=[]
    for s in samples:
        if s.box_path and s.entity_path:
            lines=s.gt_lines()
            records.append(make_record(s.id,s.labels(),predict_fields(lines,model),lines))
    if records:
        m=compute_metrics(records)
        d["field_exact_match"]={f:float(m[f]["exact_match"]) for f in ("company","date","address","total")}
        d["mean_exact_match"]=float(m["mean"]["exact_match"])
        d["all4_correct"]=sum(all(r["ok"].values()) for r in records)/len(records)
    return d

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-out",type=Path,default=MODEL_PATH)
    p.add_argument("--metrics-out",type=Path,default=ROOT/"outputs"/"ml"/"metrics.json")
    a=p.parse_args()
    splits=get_splits()
    x,y,n=flatten(splits["train"])
    if n==0 or len(set(y))<2: raise RuntimeError("Không đủ dữ liệu: cần train/img, train/box, train/entities.")
    print("Training receipts:",n,"lines:",len(y),"weak labels:",dict(Counter(y)))
    model=build_estimator()
    model.fit(x,y)
    report={
        "method":"word/char TF-IDF + LogisticRegression; weak entity-to-box matching",
        "train_receipts":n,"train_lines":len(y),"train_labels":dict(Counter(y)),
        "validation":evaluate(model,splits["val"]),
        "test":evaluate(model,splits["test"]),
        "caveat":"Metrics use original SROIE box text, not Tesseract OCR. Weak labels are noisy; test may be an internal split, not the official ICDAR benchmark."
    }
    a.model_out.parent.mkdir(parents=True,exist_ok=True)
    a.metrics_out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,a.model_out)
    a.metrics_out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("Model:",a.model_out)
    print("Metrics:",a.metrics_out)
    print("Val mean exact match:",report["validation"].get("mean_exact_match"))
    print("Test mean exact match:",report["test"].get("mean_exact_match"))
if __name__=="__main__": main()
