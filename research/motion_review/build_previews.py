"""Deterministic 3D laboratory movement previews, not a fluid/chemistry simulator."""
import json,math,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FPS=24
DURATION=8
N=FPS*DURATION+1

def smooth(a,b,t):
 u=max(0,min(1,t));return a+(b-a)*u*u*(3-2*u)
def ring(cx,y,cz,r):return [[cx+r*math.cos(i*math.tau/40),y,cz+r*math.sin(i*math.tau/40)] for i in range(41)]
def line(points,color='glass',width=2):return dict(points=points,color=color,width=width)
def vessel(cx,h,r):
 return [line(ring(cx,y,0,r)) for y in (0,h)]+[line([[cx+r*math.cos(a),0,r*math.sin(a)],[cx+r*math.cos(a),h,r*math.sin(a)]]) for a in (0,math.pi/2,math.pi,3*math.pi/2)]
def core_frame(kind,t):
 paths=[];markers=[];clearance=None
 if kind=='dropper':
  paths=vessel(0,2.4,.3)
  tip=smooth(3.5,2.8,t/2) if t<2 else 2.8 if t<6 else smooth(2.8,3.5,(t-6)/2)
  paths += [line([[0,tip,0],[0,tip+.9,0]],'tool',6)]
  phase='Approach' if t<2 else 'Hold / delivery cues' if t<6 else 'Withdraw'
  clearance=tip-2.4
  for start in (2.5,3.5,4.5):
   u=(t-start)/.65
   if 0<=u<=1:markers.append(dict(point=[0,2.8-2.1*u*u,0],color='cue',radius=.055))
 elif kind=='pour':
  paths=vessel(.65,1.25,.45)
  if t<2:x=smooth(-1.2,-.25,t/2);angle=0
  elif t<4:x=-.25;angle=smooth(0,-math.pi/3,(t-2)/2)
  elif t<6:x=-.25;angle=-math.pi/3
  else:x=smooth(-.25,-1.2,(t-6)/2);angle=smooth(-math.pi/3,0,(t-6)/2)
  cy=2.1
  def tr(p):
   a,b,c=p;b-=.5
   return [x+a*math.cos(angle)-b*math.sin(angle),cy+a*math.sin(angle)+b*math.cos(angle),c]
  paths += [line([tr(p) for p in path['points']],'tool',2) for path in vessel(0,1,.25)]
  phase='Approach receiver' if t<2 else 'Tilt' if t<4 else 'Hold pour posture' if t<6 else 'Return upright'
  if 4<=t<6:
   start=tr([0,1,0]);paths.append(line([start,[(start[0]+.65)/2,1.5,0],[.65,.3,0]],'cue',3))
 elif kind=='stir':
  paths=vessel(0,1.6,.6)
  theta=math.tau*1.5*t/8
  tip=[.22*math.cos(theta),.25,.22*math.sin(theta)]
  top=[.22*math.cos(theta),2.2,.22*math.sin(theta)]
  paths.append(line([tip,top],'tool',5))
  markers.append(dict(point=tip,color='cue',radius=.04))
  phase='Slow circular stirring'
 for axis,c in (([[0,0,0],[1,0,0]],'x'),([[0,0,0],[0,1,0]],'y'),([[0,0,0],[0,0,1]],'z')):paths.append(line(axis,c,1))
 return dict(t=round(t,4),phase=phase,paths=paths,markers=markers,tip_clearance=clearance)
KINDS=('dropper','pour','stir','pipette_aspiration','solid_transfer','paper_contact','gas_test_position','tube_shaking','tube_tilting','swirl_flask','filter_washing','sample_withdrawal','thermometer_position','bath_lowering','bath_lifting','paper_dipping','stopper_placement','apparatus_wait','electrode_placement','salt_bridge_placement','burette_delivery','burette_filling','membrane_suspension','funnel_placement')

def frame(kind,t):
 if kind in ('dropper','pour','stir'):return core_frame(kind,t)
 paths=[];marks=[];phase=kind.replace('_',' ').title();u=t/8
 if kind in ('electrode_placement','salt_bridge_placement'):
  paths=vessel(-.8,1.4,.5)+vessel(.8,1.4,.5)
  lift=smooth(1.7,0,u)
  if kind=='electrode_placement':
   for x in (-.8,.8):paths.append(line([[x,.3+lift,0],[x,1.8+lift,0]],'tool',9))
  else:
   points=[[-.8,.7+lift,0],[-.8,1.8+lift,0],[-.6,2+lift,0],[.6,2+lift,0],[.8,1.8+lift,0],[.8,.7+lift,0]]
   paths.append(line(points,'tool',7))
  return dict(t=round(t,4),phase=phase,paths=paths,markers=[],tip_clearance=None)
 if kind in ('burette_delivery','burette_filling'):
  paths=vessel(0,1.2,.45)
  paths += [line([[0,2.1,0],[0,3.9,0]],'glass',5)]
  if kind=='burette_delivery':
   paths.append(line([[-.25,2.2,0],[.25,2.2,0]],'tool',4))
   if .25<u<.75:marks.append(dict(point=[0,2.1-1.7*((u*6)%1)**2,0],color='cue',radius=.05))
  else:
   x=smooth(-1,0,min(1,u*2));paths.append(line([[x-.5,4.4,0],[x,4,0]],'tool',6))
  return dict(t=round(t,4),phase=phase,paths=paths,markers=marks,tip_clearance=None)
 if kind in ('membrane_suspension','funnel_placement'):
  paths=vessel(0,1.6,.8);lift=smooth(1.6,0,u)
  if kind=='membrane_suspension':
   paths += [line(ring(0,1.85+lift,0,.55),'tool'),line([[-.55,1.85+lift,0],[0,.3+lift,0],[.55,1.85+lift,0]],'tool')]
  else:
   paths += [line(ring(0,2+lift,0,.65),'tool'),line([[-.65,2+lift,0],[0,1.35+lift,0],[.65,2+lift,0]],'tool'),line([[0,1.35+lift,0],[0,.8+lift,0]],'tool')]
  return dict(t=round(t,4),phase=phase,paths=paths,markers=[],tip_clearance=None)
 if kind in ('tube_shaking','tube_tilting','swirl_flask'):
  theta=(.13*math.sin(math.tau*3*u) if kind=='tube_shaking' else .45*math.sin(math.pi*u) if kind=='tube_tilting' else .1*math.sin(math.tau*2*u))
  dx=.15*math.sin(math.tau*2*u) if kind=='swirl_flask' else 0
  for path in vessel(0,2.4,.3):
   points=[]
   for x,y,z in path['points']:
    y-=1.2;points.append([dx+x*math.cos(theta)-y*math.sin(theta),1.2+x*math.sin(theta)+y*math.cos(theta),z])
   paths.append(line(points,'tool',2))
 elif kind in ('bath_lowering','bath_lifting'):
  paths=vessel(0,1.6,.8);y=smooth(1.9,.25,u) if kind=='bath_lowering' else smooth(.25,1.9,u)
  for path in vessel(0,1.8,.23):paths.append(line([[x,y+py,z] for x,py,z in path['points']],'tool',2))
 elif kind in ('paper_contact','paper_dipping'):
  if kind=='paper_contact':
   paths=[line([[-.8,.1,-.15],[.8,.1,-.15],[.8,.1,.15],[-.8,.1,.15],[-.8,.1,-.15]],'glass')]
   tip=smooth(.9,.12,u*2) if u<.5 else smooth(.12,.9,(u-.5)*2)
   paths.append(line([[0,tip,0],[.7,tip+1,0]],'tool',4))
  else:
   paths=vessel(0,1.6,.6);y=smooth(2.2,.6,u*2) if u<.5 else smooth(.6,2.2,(u-.5)*2)
   paths.append(line([[-.12,y,0],[.12,y,0],[.12,y+.8,0],[-.12,y+.8,0],[-.12,y,0]],'tool',3))
 else:
  paths=vessel(0,2.4,.3)
  if kind=='apparatus_wait':pass
  elif kind=='stopper_placement':
   y=smooth(3.3,2.4,u);paths.append(line(ring(0,y,0,.32),'tool',5))
  elif kind=='gas_test_position':
   x=smooth(1.2,.42,u*2) if u<.5 else smooth(.42,1.2,(u-.5)*2)
   paths.append(line([[x,2.5,0],[x+.8,2.65,0]],'tool',4))
  elif kind in ('pipette_aspiration','sample_withdrawal','thermometer_position'):
   y=smooth(3.2,.7,u*2) if u<.5 else smooth(.7,3.2,(u-.5)*2)
   if kind=='thermometer_position':y=smooth(3.2,.7,min(1,u*2))
   paths.append(line([[0,y,0],[0,y+2,0]],'tool',4))
  elif kind=='solid_transfer':
   x=smooth(-1,0,min(1,u*3));paths.append(line([[x,2.9,0],[x-.8,3.1,0]],'tool',5));paths.append(line(ring(x,2.9,0,.12),'tool',3))
   if .4<u<.6:marks.append(dict(point=[0,2.9-2*(u-.4)/.2,0],color='cue',radius=.05))
  elif kind=='filter_washing':
   # Cone is a motion target only, not a complete filtration apparatus.
   paths=[line(ring(0,1.8,0,.65)),line([[-.65,1.8,0],[0,.6,0],[.65,1.8,0]])]
   x=.2*math.sin(math.tau*u);paths.append(line([[x,2.25,0],[x,3.2,0]],'tool',4))
   if .25<u<.75:paths.append(line([[x,2.25,0],[x,1.1,0]],'cue',2))
 return dict(t=round(t,4),phase=phase,paths=paths,markers=marks,tip_clearance=None)

def build():
 result={'schema_version':1,'status':'draft_motion_review_only','fps':FPS,'duration_seconds':DURATION,'units':'illustrative_scene_units_not_mm','limitations':['No fluid simulation, dose calibration, hand rig or chemical reaction.','Timing is a proposed presentation motion, not a measured procedure.','Not a model-ready depth/pose conditioning export.','Each loop is an independent preview, not a continuous repeated experiment.'], 'motions':{}}
 for kind in KINDS:
  frames=[frame(kind,i/FPS) for i in range(N)]
  assert all(math.isfinite(v) for f in frames for p in f['paths'] for pt in p['points'] for v in pt)
  if kind=='dropper':assert min(f['tip_clearance'] for f in frames)>=.399
  if kind=='stir':assert all(math.hypot(f['markers'][0]['point'][0],f['markers'][0]['point'][2])<.6 for f in frames)
  result['motions'][kind]=frames
 # Keep static geometry data compact enough for an offline browser.
 def rounded(v):
  if isinstance(v,float):return round(v,4)
  if isinstance(v,list):return [rounded(x) for x in v]
  if isinstance(v,dict):return {k:rounded(x) for k,x in v.items()}
  return v
 result=rounded(result)
 (ROOT/'motion_templates.json').write_text(json.dumps(result,separators=(',',':')))
 return result

def render(data):
 from PIL import Image,ImageDraw
 from imageio_ffmpeg import get_ffmpeg_exe
 colors={'glass':'#9caec7','tool':'#eac27b','cue':'#68d7cb','x':'#ef989c','y':'#9bd4ab','z':'#9cb9ef'}
 def project(p):
  x,y,z=p;yaw=.48;depth=x*math.sin(yaw)+z*math.cos(yaw)
  return 480+155*(x*math.cos(yaw)-z*math.sin(yaw)),470-115*y+45*depth
 previews=ROOT/'previews';previews.mkdir(exist_ok=True)
 for kind,frames in data['motions'].items():
  output=previews/(kind+'.mp4')
  proc=subprocess.Popen([get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r',str(FPS),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',str(output)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  try:
   for f in frames:
    image=Image.new('RGB',(960,540),'#142130');d=ImageDraw.Draw(image)
    for path in f['paths']:d.line([project(p) for p in path['points']],fill=colors[path['color']],width=path['width'])
    for mark in f['markers']:
     x,y=project(mark['point']);d.ellipse((x-5,y-5,x+5,y+5),fill=colors[mark['color']])
    d.text((25,20),kind.upper()+' - '+f['phase'],fill='white');d.text((25,42),f"{f['t']:.2f}s / 8s | Draft motion only; illustrative units",fill='#bbcbdb')
    d.text((25,510),'Teal = illustrative delivery/path cue. No fluid, dose or reaction simulation.',fill='#bbcbdb')
    if abs(f['t']-4)<.01:image.save(previews/(kind+'.png'))
    proc.stdin.write(image.tobytes())
  finally:
   proc.stdin.close()
  err=proc.stderr.read();assert proc.wait()==0,err.decode()
  print('Rendered',output)
if __name__=='__main__':render(build())
