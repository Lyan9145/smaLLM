"""Read-only snapshot for a detached 15-minute check."""
import json,sys,time,subprocess
from pathlib import Path
out=Path(sys.argv[1]); delay=int(sys.argv[2]) if len(sys.argv)>2 else 0
if delay: time.sleep(delay)
report={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'complete':(out/'COMPLETE').exists(),
        'failure':(out/'FAILED').read_text() if (out/'FAILED').exists() else None,'runs':[]}
for p in sorted(out.glob('*/metrics.json')):
 d=json.loads(p.read_text());report['runs'].append({'name':p.parent.name,'bpb':d['validation']['bpb'],
 'targets':d['processed_targets'],'selected_weights':d['selected_weights'],'seconds':d['train_seconds']})
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
report['latest_progress']={p.name:p.read_text()[-1500:] for p in out.glob('*.log') if p.name!='check-15min.log'}
(out/'status-15min.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
