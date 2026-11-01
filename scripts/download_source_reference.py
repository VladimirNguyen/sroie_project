import json,pathlib,urllib.request,concurrent.futures,random,time
root=pathlib.Path(__file__).resolve().parents[1];sha='27be4271b251c256f695acbade9a801bffe85994';tree=json.load(urllib.request.urlopen('https://api.github.com/repos/metatatt/sorie/git/trees/'+sha+'?recursive=1'));files=[x for x in tree['tree'] if x['path'].startswith('data/') and x['type']=='blob'];ids=sorted({pathlib.Path(x['path']).stem for x in files});random.Random(42).shuffle(ids);ids=ids[:300];split={s:('test' if i<200 else 'train') for i,s in enumerate(ids)}
def get(x):
 s=pathlib.Path(x['path']).stem
 if s not in split:return
 kind=x['path'].split('/')[1];dst=root/'data/raw/SROIE2019'/split[s]/{'img':'img','key':'entities','box':'box'}[kind]/(s+(pathlib.Path(x['path']).suffix if kind=='img' else '.txt'));dst.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try: data=urllib.request.urlopen('https://raw.githubusercontent.com/metatatt/sorie/'+sha+'/'+x['path'],timeout=45).read();dst.write_bytes(data);return
  except Exception:
   if attempt==3:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as e:
 for i,_ in enumerate(e.map(get,files)):
  if i%200==0: print(i,flush=True)
(root/'docs/source_manifest.json').write_text(json.dumps({'repository':'metatatt/sorie','commit':sha,'access_date':'2026-11-01','selection_seed':42,'selected':split},indent=2))
print('DONE',flush=True)
