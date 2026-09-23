"""Validation-only scale/regularization search, CPU gate, then two long seeds."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
out=a.output; out.mkdir(parents=True,exist_ok=True)
python=sys.executable

def run(args, name):
 cmd=[python,'-u',*map(str,args)]
 with (out/'commands.jsonl').open('a') as f: f.write(json.dumps({'time':time.time(),'argv':cmd})+'\n')
 print(json.dumps({'started':name,'time':time.time()}),flush=True)
 with (out/(name+'.log')).open('x') as f: subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)

def train(name,config,lr,ema,updates=916,seed=17):
 run(['train.py','--implementation','student','--config',config,'--device','cuda','--precision','bf16','--tf32',
      '--threads',4,'--seed',seed,'--micro-batch-size',32,'--grad-accum',4,'--updates',updates,
      '--lr',lr,'--warmup',round(updates*.04),'--min-lr-ratio',.1,'--betas',.9,.95,
      '--weight-decay',.1,'--ema',ema,'--eval-every',100 if updates==916 else 305,
      '--save-every',100,'--log-every',25,'--run-log',out/'RUN_LOG.csv','--run-dir',out/name],name)
 d=json.loads((out/name/'metrics.json').read_text())
 run(['evaluate.py','--checkpoint',out/name/'checkpoint.pt','--device','cpu','--precision','fp32',
      '--threads',4,'--split','validation'],name+'-cpu')
 cpu=json.loads((out/name/'validation_cpu_fp32.json').read_text())
 assert abs(cpu['bpb']-d['validation']['bpb'])<1e-5
 return {'name':name,'config':str(config),'lr':lr,'ema':ema,'seed':seed,'bpb':cpu['bpb'],
         'cpu_seconds':cpu['seconds'],'assets':d['serialized_inference_asset_bytes'],
         'selected_step':d['selected_step'],'selected_weights':d['selected_weights']}

try:
 run(['-m','unittest','discover','-s','tests','-v'],'tests')
 run(['evaluate.py','--checkpoint','/root/autodl-tmp/mp1-phase1-20260920/baseline-s17/checkpoint.pt',
      '--device','cpu','--precision','fp32','--threads',4,'--split','validation','--output',out/'baseline_cpu.json'],'baseline-cpu')
 baseline=json.loads((out/'baseline_cpu.json').read_text())
 results=[]
 candidates=[('ema-control','student_matched_no_local',0,.002,.99),
             ('scale8-lr1','student_8',.05,.001,.99),
             ('scale8-lr2','student_8',.05,.002,.99),
             ('scale8-drop10','student_8',.1,.001,.99),
             ('scale10-lr1','student_10',.05,.001,.99)]
 for name,base,drop,lr,ema in candidates:
  config=out/'configs'/(name+'.json');config.parent.mkdir(exist_ok=True)
  data=json.loads(Path('configs',base+'.json').read_text());data.update(local_mixing=False,dropout=drop)
  config.write_text(json.dumps(data,indent=2)+'\n')
  row=train(name,config,lr,ema);row['eligible']=row['cpu_seconds']<=min(24,4*baseline['seconds']) and row['assets']<56*2**20
  results.append(row);(out/'search_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
 eligible=[r for r in results if r['eligible']]
 if not eligible: raise RuntimeError('No candidate passes CPU/asset gate')
 winner=min(eligible,key=lambda r:(r['bpb'],r['assets']))
 (out/'selection.json').write_text(json.dumps(winner,indent=2)+'\n')
 for seed in [17,42]:
  row=train('long-s'+str(seed),winner['config'],winner['lr'],winner['ema'],3052,seed)
  results.append(row);(out/'search_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
 (out/'COMPLETE').write_text('Validation-only search and two long seeds complete. Review before freezing/testing.\n')
except BaseException as e:
 (out/'FAILED').write_text(repr(e)+'\n');raise
