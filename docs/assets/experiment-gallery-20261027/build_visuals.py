from pathlib import Path
import json, math, hashlib, subprocess, urllib.request
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).parent
SRC=ROOT/'.source-cache'; OUT=ROOT; SRC.mkdir(exist_ok=True)
VID=ROOT/'.video-cache'; VID.mkdir(exist_ok=True)
BASE='https://raw.githubusercontent.com/leonyoon-3dai/behavior-submission-2026/409aa8bde8455e742bc5f4d087437fedc5d1565c/'
FILES={
 'verified-facts.json':'docs/reports/submission-2026/presentation-assets/verified-facts.json',
 'results.csv':'results/results.csv',
 'rollout-source-provenance.json':'docs/reports/submission-2026/presentation-assets/rollout-source-provenance.json',
}
for name in ['task001-public0-t060-full.png','task001-public0-t132-full.png','task001-public0-t237-full.png','task002-public0-t300-head.png','task005-public1-last-head.png']:
 FILES[name]='docs/reports/submission-2026/presentation-assets/'+name
for name,path in FILES.items():
 dest=SRC/name
 if not dest.exists():dest.write_bytes(urllib.request.urlopen(BASE+path,timeout=120).read())
vp=json.loads((SRC/'rollout-source-provenance.json').read_text())
original_videos=[]
for name in ['task001-public0-t060-full.png','task002-public0-t300-head.png','task005-public1-last-head.png']:
 item=next(x for x in vp['items'] if x['file']==name)
 dest=VID/('task'+item['task']+'-public'+str(item['public_index'])+'.mp4')
 if not dest.exists():dest.write_bytes(urllib.request.urlopen(item['source_video_url'],timeout=120).read())
 digest=hashlib.sha256(dest.read_bytes()).hexdigest()
 assert digest==item['source_video_sha256'],str(dest)
 original_videos.append({'file':dest.name,'sha256':digest,'size':dest.stat().st_size})
F=json.loads((SRC/'verified-facts.json').read_text())
W=1600
BG='#F7F9FC'; FG='#172537'; MUTED='#607087'; BLUE='#2878C8'; TEAL='#2A938F'; ORANGE='#C07A3F'; LINE='#D9E2EC'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def f(n,b=False): return ImageFont.truetype(BOLD if b else FONT,n)
def base(h,title,sub):
 im=Image.new('RGB',(W,h),BG);d=ImageDraw.Draw(im)
 d.text((48,35),title,font=f(42,True),fill=FG);d.text((50,96),sub,font=f(23),fill=MUTED)
 return im,d
def img(im,name,box,head=False):
 a=Image.open(SRC/name).convert('RGB')
 if head and a.width==672:a=a.crop((224,0,672,448))
 x,y,w,h=box;a.thumbnail((w,h),Image.Resampling.LANCZOS)
 im.paste(a,(int(x+(w-a.width)/2),int(y+(h-a.height)/2)))
def textlines(d,xy,lines,size=25,color=FG,bold=False,dy=None):
 x,y=xy
 for line in lines:d.text((x,y),line,font=f(size,bold),fill=color);y+=dy or size+11
def card(d,box,fill='white'):
 x,y,w,h=box;d.rounded_rectangle((x,y,x+w,y+h),radius=18,fill=fill,outline=LINE,width=2)
def arrow(d,a,b):
 d.line((a,b),fill=MUTED,width=4)
 angle=math.atan2(b[1]-a[1],b[0]-a[0]);r=13
 pts=[b,(b[0]-r*math.cos(angle-.5),b[1]-r*math.sin(angle-.5)),(b[0]-r*math.cos(angle+.5),b[1]-r*math.sin(angle+.5))]
 d.polygon(pts,fill=MUTED)
def save(im,name):
 if name.endswith('.jpg'):im.save(OUT/name,quality=84,optimize=True)
 else:im.save(OUT/name,optimize=True)

# Three original head-camera crops. Labels separate observed behavior from final goals.
im,d=base(800,'A robot policy in three household tasks','Original final-panel recordings from Leonard Yoon\'s evaluation runs')
cards=[('task001-public0-t132-full.png','001  Pick up trash','Public 0   /   video 132 s','Final goal progress 0.0000'),('task002-public0-t300-head.png','002  Put away decorations','Public 0   /   video 300 s','Final goal progress 0.5714'),('task005-public1-last-head.png','005  Set mousetraps','Public 1   /   video 509.8 s','Final goal progress 0.5000')]
for j,(file,title,stamp,q) in enumerate(cards):
 x=48+j*510;card(d,(x,150,485,550));img(im,file,(x+18,168,449,370),head=True)
 textlines(d,(x+20,554),[title,stamp,q],size=23,dy=40)
d.text((50,735),'12 final rollouts     3 tasks     0 complete successes     Simulation only',font=f(26,True),fill=FG)
save(im,'experiment-overview.jpg')

# Readable policy input / output contract.
im,d=base(830,'The observation-to-action interface','The submitted policy consumes RGB and proprioception. Depth is discarded at the policy boundary.')
img(im,'task001-public0-t060-full.png',(48,170,780,520))
d.text((50,716),'Recorded views: two wrists at left, head camera at right',font=f(23),fill=MUTED)
card(d,(880,160,670,160));textlines(d,(908,182),['3 RGB cameras + 61D state','Shape, dtype and finite-value checks','Task ID routes one active policy'],size=26,dy=40)
arrow(d,(1215,325),(1215,370))
card(d,(880,380,670,150));textlines(d,(908,400),['WebSocket / msgpack policy server','Episode reset and model switching','Asynchronous action generation'],size=26,dy=39)
arrow(d,(1215,537),(1215,580))
card(d,(880,590,670,140));textlines(d,(908,612),['23D finite action','R1Pro execution in OmniGibson','State dimension does not mean robot DoF'],size=25,dy=36)
save(im,'policy-interface.jpg')

im,d=base(850,'Frozen actor, learned action correction','Task-specific auxiliary training: 20 official demonstrations and 1,000 updates per task')
nodes=[(48,180,390,155,'Frozen Ilia / pi0.5',['Released pretrained weights','Two base action candidates']),
 (585,180,400,155,'Residual action editor',['Keep two base candidates','Add one corrected candidate']),
 (1130,180,420,155,'Value-based selection',['Score all three candidates','Use twin target-critic minimum'])]
for x,y,w,h,title,ls in nodes:
 card(d,(x,y,w,h));d.text((x+20,y+20),title,font=f(26,True),fill=FG);textlines(d,(x+20,y+66),ls,size=23,dy=34)
arrow(d,(440,255),(577,255));arrow(d,(987,255),(1122,255))
card(d,(810,440,740,160));textlines(d,(835,461),['Retained execution wrapper','Stage voting and correction rules remain active','Rules can modify the selected action'],size=25,dy=39)
arrow(d,(1340,338),(1340,430))
card(d,(48,440,610,160));textlines(d,(72,461),['23D action execution','Execute 15 actions, carry 4 actions','Prefetch the next chunk asynchronously'],size=25,dy=39)
arrow(d,(802,520),(670,520))
d.rounded_rectangle((48,660,1550,790),radius=18,fill='#E8EEF7')
textlines(d,(76,683),['Actor weights stay frozen. Encoder, critics and edit actor receive the auxiliary updates.',
 'Learned critic values select actions. Official final goal progress measures task outcomes.'],size=26,dy=44)
save(im,'adaptation-and-execution.png')

# Measured final values, full 0..1 range, exact labels, no success-rate ambiguity.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.7),sharey=True)
names={2:'002  Put away decorations',1:'001  Pick up trash',5:'005  Set mousetraps'}
for ax,row in zip(axes,F['final']):
 vals=row['Q_by_instance'];bars=ax.bar(range(4),vals,color=BLUE,width=.65)
 ax.set_ylim(0,1.08);ax.set_xticks(range(4),['P0','P1','P2','P3']);ax.set_title(names[row['task_id']],loc='left',fontsize=16,pad=15)
 ax.set_xlabel('Public instance');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 for j,v in enumerate(vals):ax.text(j,v+.025,f'{v:.3f}',ha='center',fontsize=13)
axes[0].set_ylabel('Final goal progress (0 to 1)')
fig.suptitle('Final evaluation: 12 rollouts, 0 complete successes',x=.055,ha='left',fontsize=22,fontweight='bold')
fig.text(.055,.025,'Mean final goal progress 0.224206. RT only; no matched A0 control. Self-evaluation, not a leaderboard score.',fontsize=12,color=MUTED)
fig.subplots_adjust(top=.75,bottom=.21,left=.055,right=.98,wspace=.2);fig.savefig(OUT/'final-per-instance.png',dpi=130,facecolor='white');plt.close(fig)

fig,ax=plt.subplots(figsize=(15,5.5))
pilot=F['pilot'];x=np.arange(6);width=.34
a=[v['A0_mean_Q'] for v in pilot];b=[v['RT_mean_Q'] for v in pilot]
ax.bar(x-width/2,a,width,label='Original A0',color='#8194AB');ax.bar(x+width/2,b,width,label='A0 + RT auxiliaries',color=BLUE)
ax.set_xticks(x,['Turn on\nradio','Pick up\ntrash','Put away\ndecorations','Clean plates\nand food','Set\nmousetraps','Pick up\ntoys'])
ax.set_ylim(0,1.08);ax.set_ylabel('Mean final goal progress');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right')
for xx,aa,bb in zip(x,a,b):
 ax.text(xx-width/2,aa+.025,f'{aa:.3f}',ha='center',fontsize=11);ax.text(xx+width/2,bb+.025,f'{bb:.3f}',ha='center',fontsize=11)
fig.suptitle('Development comparison: 6 tasks, 36 rollouts',x=.07,ha='left',fontsize=22,fontweight='bold')
fig.text(.07,.04,'Overall mean: 0.455 to 0.329    Complete successes: 5/18 to 1/18    No overall performance improvement.',fontsize=13,color=FG)
fig.subplots_adjust(top=.8,bottom=.23,left=.07,right=.98);fig.savefig(OUT/'development-comparison.png',dpi=130,facecolor='white');plt.close(fig)

im,d=base(780,'Object handling and placement are different outcomes','Task 001 / public 0. Each panel preserves all three recorded camera views.')
for i,(fn,t,label) in enumerate([('task001-public0-t060-full.png','60 s','Early motion'),('task001-public0-t132-full.png','132 s','Trash can near right hand'),('task001-public0-t237-full.png','237 s','Late pose persists')]):
 x=48+i*510;card(d,(x,160,486,445));img(im,fn,(x+12,175,462,308));textlines(d,(x+18,510),[t+' video time',label],size=24,dy=42)
d.rounded_rectangle((48,640,1550,738),radius=16,fill='#E8EEF7');textlines(d,(72,655),['Final goal progress 0.0000. The task requires each can inside the trash can.',
 'Frames support visual review; exact contacts and physical root causes remain unverified.'],size=24,dy=35)
save(im,'task001-storyboard.jpg')

im,d=base(620,'A reproducible submission package','Policy code, model identity, runtime and original outputs remain connected')
items=[('Code + config',['Policy adapter','Robot / wrapper revision']),('Model manifest',['Frozen actor identity','Task-specific auxiliaries']),('Docker image',['Embedded weights','Anonymous digest pull']),('Evaluation',['12 original JSON files','12 unchanged MP4 videos'])]
for i,(title,ls) in enumerate(items):
 x=48+i*390;card(d,(x,195,345,220));d.text((x+22,220),title,font=f(27,True),fill=FG);textlines(d,(x+22,275),ls,size=21,dy=39)
 if i<3:arrow(d,(x+347,305),(x+381,305))
d.text((50,460),'Checksums + pinned revisions + public provenance',font=f(31,True),fill=BLUE)
textlines(d,(50,515),['Submission accepted on 11 Oct 2026. Organizer performance verification remains pending.',
 'Deployment smoke checks use synthetic inputs and do not measure household-task success.'],size=23,dy=37)
save(im,'reproducibility.png')

# Preview clips are temporal excerpts of verified original files. No new simulator runs.
clips=[('task001-public0',60,6),('task002-public0',292,6),('task005-public1',100,6)]
for name,start,duration in clips:
 filters='crop=448:448:224:0,fps=2,scale=224:224:flags=lanczos,split[a][b];[a]palettegen=max_colors=24[p];[b][p]paletteuse=dither=bayer:bayer_scale=4'
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-t',str(duration),'-i',str(VID/f'{name}.mp4'),'-filter_complex',filters,'-loop','0',str(OUT/f'{name}-preview.tmp.gif')],check=True)
 (OUT/f'{name}-preview.tmp.gif').replace(OUT/f'{name}-preview.gif')

manifest={'source_commit':'409aa8bde8455e742bc5f4d087437fedc5d1565c','created_date':'2026-10-27','source_notes':'Original participant evaluation frames. Charts use verified-facts.json and results.csv. No evaluation or policy changes.','assets':[]}
for path in sorted(p for p in OUT.iterdir() if p.suffix in ('.png','.jpg','.gif')):
 manifest['assets'].append({'file':path.name,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
manifest['original_videos']=original_videos
manifest['video_excerpts']=[{'file':f'{n}-preview.gif','original':f'{n}.mp4','start_video_seconds':s,'duration_video_seconds':d,'crop':[224,0,672,448],'fps':2,'display_size':[224,224],'note':'Head camera only; sampled playback excerpt, not wall-clock latency'} for n,s,d in clips]
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
