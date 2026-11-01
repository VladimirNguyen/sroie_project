"""Reproducible OCR experiment. Never alter labels or fit on test."""
import os
os.environ.setdefault('OMP_THREAD_LIMIT','1')
import sys,json,hashlib,time,platform,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data_loader import get_splits
from src.preprocess import read_image
from src.pipeline import run_ocr
from src.ocr_engine import get_engine
from src.extractors import EXTRACTORS
from src.evaluate import make_record,compute_metrics
from src.normalize import NORMALIZERS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiment';OUT.mkdir(parents=True,exist_ok=True)
def work(s):
 engine=get_engine('tesseract',psm=6);rows=[]
 for variant in ('basic','improved'):
  key=hashlib.sha256(s.image_path.read_bytes()+f'{variant}:psm6:eng:v1'.encode()).hexdigest()
  dest=OUT/'ocr'/f'{s.split}_{s.id}_{key}.json';dest.parent.mkdir(exist_ok=True)
  if dest.exists(): o=json.loads(dest.read_text())
  else:
   lines,proc,sec=run_ocr(read_image(s.image_path),engine,variant)
   o={'id':s.id,'split':s.split,'variant':variant,'seconds':sec,'image_sha256':hashlib.sha256(s.image_path.read_bytes()).hexdigest(),'lines':[l.to_dict() for l in lines]}
   dest.write_text(json.dumps(o,ensure_ascii=False),encoding='utf-8')
  text=[l['text'] for l in o['lines']]
  for name,fun in EXTRACTORS.items():
   pred=fun(text);r=make_record(s.id,s.labels(),pred,text)
   r.update(config=f'{variant}/{name}',split=s.split,seconds=o['seconds'],ocr=text,normalized={f:NORMALIZERS[f](v) for f,v in pred.items()})
   rows.append(r)
 return rows
if __name__=='__main__':
 splits=get_splits();manifest={k:[s.id for s in v] for k,v in splits.items()}
 (OUT/'splits.json').write_text(json.dumps(manifest,indent=2))
 hashes={}
 for k,ss in splits.items():
  for s in ss:
   h=hashlib.sha256(s.image_path.read_bytes()).hexdigest()
   if h in hashes and hashes[h]!=k:raise RuntimeError('Duplicate image across splits')
   hashes[h]=k
 records=[];metrics=[]
 for split in ('val','test'):
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for i,rr in enumerate(pool.map(work,splits[split])):
    records.extend(rr)
    if i%10==0:print(split,i,flush=True)
  for config in ('basic/baseline','basic/improved','improved/baseline','improved/improved'):
   rs=[r for r in records if r['split']==split and r['config']==config];m=compute_metrics(rs)
   row={'split':split,'config':config,'n':len(rs),'EM_mean':m['mean']['exact_match'],'F1_mean':m['mean']['f1'],'all4':sum(all(r['ok'].values()) for r in rs)/len(rs),'seconds':sum(r['seconds'] for r in rs)/len(rs)}
   row.update({f'EM_{f}':m[f]['exact_match'] for f in NORMALIZERS});metrics.append(row)
  (OUT/'predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
  pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
 flat=[]
 for r in records:
  for f in NORMALIZERS:flat.append({'id':r['id'],'split':r['split'],'config':r['config'],'field':f,'truth':r['gt'][f],'prediction':r['pred'][f],'correct':r['ok'][f],'error_type':r['error_type'][f]})
 pd.DataFrame(flat).to_csv(OUT/'per_field.csv',index=False)
 (OUT/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'tesseract':str(get_engine().pt.get_tesseract_version()),'psm':6,'lang':'eng','workers':4},indent=2))
 print(pd.DataFrame(metrics).to_string(index=False),flush=True)
