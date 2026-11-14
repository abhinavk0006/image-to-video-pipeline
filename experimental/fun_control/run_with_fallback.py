"""Fresh-process OOM-only retries; every attempted profile is preserved."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def sizes(width,height,fallbacks):
 result=[(width,height)]
 for text in fallbacks:
  w,h=map(int,text.lower().split('x'))
  if min(w,h)<=0 or w%16 or h%16:raise ValueError('Fallback dimensions must be positive multiples of 16')
  if w*h>=result[-1][0]*result[-1][1]:raise ValueError('Fallback areas must strictly descend')
  if w*height!=h*width:raise ValueError('Keep the reference/control aspect ratio fixed')
  result.append((w,h))
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=768);p.add_argument('--height',type=int,default=432);p.add_argument('--fallback-size',action='append',default=[]);p.add_argument('--dry-run',action='store_true');a,other=p.parse_known_args()
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);report=root/('suite_preflight.json' if a.dry_run else 'suite.json')
 if report.exists() and not a.dry_run:raise FileExistsError('Use a fresh suite output directory')
 record={'scientifically_verified':False,'accepted_seconds':None,'attempts':[]};started=time.perf_counter();status=1
 for i,(w,h) in enumerate(sizes(a.width,a.height,a.fallback_size)):
  out=root/f'attempt_{i:02d}_{w}x{h}'
  command=[sys.executable,str(Path(__file__).with_name('run_control.py')),'--output',str(out),'--width',str(w),'--height',str(h)]+other+(['--dry-run'] if a.dry_run else [])
  before=time.perf_counter();result=subprocess.run(command)
  manifest_path=out/('preflight.json' if a.dry_run else 'run.json');manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
  record['attempts'].append({'size':[w,h],'returncode':result.returncode,'wall_seconds':time.perf_counter()-before,'cuda_oom_detected':manifest.get('cuda_oom_detected',False),'manifest':str(manifest_path)})
  record['wall_seconds']=time.perf_counter()-started;report.write_text(json.dumps(record,indent=2))
  status=result.returncode
  if status==0 or not manifest.get('cuda_oom_detected',False) or a.dry_run:break
 return status
if __name__=='__main__':raise SystemExit(main())
