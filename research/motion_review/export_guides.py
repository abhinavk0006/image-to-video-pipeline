"""Export draft synthetic line guides in four 2-second chunks per motion."""
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
from imageio_ffmpeg import get_ffmpeg_exe
from build_previews import KINDS,frame
ROOT=Path(__file__).resolve().parent

def main(kinds=KINDS):
 out=ROOT/'guides';out.mkdir(exist_ok=True)
 def project(p):return (384+110*(p[0]*.887-p[2]*.461),380-85*p[1]+32*(p[0]*.461+p[2]*.887))
 for kind in kinds:
  for chunk in range(4):
   path=out/f'{kind}_chunk_{chunk+1:02d}.mp4'
   proc=subprocess.Popen([get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','768x432','-r','16','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   for i in range(33):
    img=Image.new('RGB',(768,432),'black');d=ImageDraw.Draw(img)
    for p in frame(kind,chunk*2+i/16)['paths']:
     if p['color'] not in ('glass','tool'):continue
     d.line([project(pt) for pt in p['points']],fill='white',width=2)
    proc.stdin.write(img.tobytes())
   proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err.decode()
 print('Exported',len(kinds)*4,'draft synthetic line chunks; no calibration/chemistry approval')
if __name__=='__main__':main()
