#!/usr/bin/env python3
"""Build the English visual overview from preserved results and rollout frames.

Run from any directory: python scripts/build_visual_presentation.py
Requires reportlab and Pillow. All charts use frozen measured values.
"""
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reports/submission-2026'
ASSETS = OUT / 'presentation-assets'
PDF = OUT / 'presentation-pilot6-submission12.pdf'
FACTS = json.loads((ASSETS / 'verified-facts.json').read_text())
FINAL = json.loads((ROOT / 'results/results.json').read_text())
W, H = 960, 540
NAVY, TEAL, ORANGE = '#10263A', '#087F83', '#B95332'
WHITE, PAPER, MUTED, LINE = '#FFFFFF', '#F8FAFB', '#546775', '#DCE4E9'
GITHUB = 'https://github.com/leonyoon-3dai/behavior-submission-2026'
BASE = GITHUB + '/blob/main/'
FONTDIR = Path('/usr/share/fonts/truetype/dejavu')
for name, filename in [('Regular','DejaVuSans.ttf'), ('Bold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONTDIR / filename)))
c = canvas.Canvas(str(PDF), pagesize=(W,H), pageCompression=1)
c.setTitle('Ilia + EXPO-RT: household robot experiments and final submission')
c.setAuthor('Leonard Yoon')
c.setSubject('Visual overview of the completed BEHAVIOR 2026 development pilot and partial submission')
PAGE = 0
TEXT_BOXES = []

def rect(x,y,w,h,color,stroke=None):
    c.setFillColor(HexColor(color))
    if stroke: c.setStrokeColor(HexColor(stroke))
    c.rect(x,H-y-h,w,h,fill=1,stroke=bool(stroke))

def line(x1,y1,x2,y2,color=LINE,width=1):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width)
    c.line(x1,H-y1,x2,H-y2)

def text(x,y,s,size=18,color=NAVY,bold=False,maxw=None,leading=None):
    font='Bold' if bold else 'Regular';leading=leading or size*1.25
    lines=[]
    for seg in s.split('\n'):
        cur=''
        for word in seg.split(' '):
            test=(cur+' '+word).strip()
            if maxw and cur and pdfmetrics.stringWidth(test,font,size)>maxw:
                lines.append(cur);cur=word
            else:cur=test
        lines.append(cur)
    c.setFillColor(HexColor(color));c.setFont(font,size)
    for i,t in enumerate(lines):
        width=pdfmetrics.stringWidth(t,font,size); top=y+i*leading
        assert x+width<=W-20,(PAGE,t,'right overflow',x+width)
        assert top+size<=H-12,(PAGE,t,'bottom overflow')
        c.drawString(x,H-top-size,t)
        TEXT_BOXES.append({'page':PAGE,'text':t,'bbox':[x,top,x+width,top+size]})
    return len(lines)*leading

def image(name,x,y,w,h):
    """Place a preserved camera crop without further cropping or distortion."""
    p=ASSETS/name;im=Image.open(p);ratio=min(w/im.width,h/im.height)
    dw,dh=im.width*ratio,im.height*ratio
    encoded=BytesIO()
    im.convert('RGB').save(encoded,format='JPEG',quality=94,subsampling=0)
    encoded.seek(0)
    c.drawImage(ImageReader(encoded),x+(w-dw)/2,H-y-dh,width=dw,height=dh)
    return dw,dh

def link(x,y,label,url,size=11,color=TEAL):
    text(x,y,label,size,color)
    width=pdfmetrics.stringWidth(label,'Regular',size)
    c.linkURL(url,(x,H-y-size*1.3,x+width,H-y+2),relative=0,thickness=0)

def page(title=None,sub=None,dark=False,source=None):
    global PAGE
    if PAGE:c.showPage()
    PAGE+=1
    rect(0,0,W,H,NAVY if dark else PAPER)
    if title:text(44,35,title,31,WHITE if dark else NAVY,True)
    if sub:text(44,82,sub,16,'#CAD6DE' if dark else MUTED,maxw=865)
    color='#AFC2CF' if dark else MUTED
    text(44,512,'robogoomy / BEHAVIOR 2026',9,color)
    text(887,510,f'{PAGE:02d}',11,color,True)
    if source:link(360,512,'Evidence',BASE+source,9,color)

def arrow(x1,y1,x2,y2,color=MUTED):
    line(x1,y1,x2,y2,color,2)
    if y1==y2:
        line(x2-7,y2-4,x2,y2,color,2);line(x2-7,y2+4,x2,y2,color,2)
    else:
        line(x2-4,y2-7,x2,y2,color,2);line(x2+4,y2-7,x2,y2,color,2)

def node(x,y,w,h,label,fill=PAPER,stroke=LINE,fs=19):
    rect(x,y,w,h,fill,stroke)
    lines=label.split('\n')
    for i,t in enumerate(lines):
        tw=pdfmetrics.stringWidth(t,'Bold',fs)
        text(x+(w-tw)/2,y+(h-len(lines)*fs*1.25)/2+i*fs*1.25,t,fs,NAVY,True)

# 01: One question and one real scene.
page(dark=True,source='results/results.json')
image('task002-public0-t300-head.png',425,38,492,444)
text(44,42,'ILIA + EXPO-RT',13,'#88DDDA',True)
text(44,111,'Small action edits\nfor household\nrobots',37,WHITE,True,leading=45)
text(44,282,'Can learned corrections\nhelp complete daily tasks?',20,'#C7D6E0',leading=29)
text(44,392,'Leonard Yoon',18,WHITE,True)
text(44,423,'Independent researcher\nChallenge submission / October 2026',13,'#C7D6E0',leading=21)
text(565,486,'Actual rollout / task 002',10,'#C7D6E0')

# 02: Show the task domain using the source deck's actual demonstrations.
page('Six everyday tasks','Human demonstrations show the behaviors we want the robot to learn.')
names=[('000','Radio'),('001','Trash'),('002','Decorations'),('003','Plates and food'),('005','Mousetraps'),('007','Toys')]
for i,(tid,name) in enumerate(names):
    x=44+(i%3)*300;y=127+(i//3)*183
    image(f'demo-task{tid}.jpg',x,y,161,161)
    text(x+174,y+47,tid,12,TEAL,True)
    text(x+174,y+72,name,18,NAVY,True,maxw=118,leading=23)
text(44,488,'Source: original human demonstration panels from the previous presentation.',10,MUTED)

# 03: A schematic method overview, with no invented scores or trajectories.
page('Ilia proposes actions. EXPO-RT edits them.','The original robot policy stays frozen.',source='docs/submission-2026/ILIA-RT-EXPO-INFERENCE-AUDIT.md')
node(44,153,247,76,'Ilia policy\nExisting skills',fill='#E7EDF2')
node(44,311,247,76,'Small correction\nLearned for each task',fill='#E0F2EE')
arrow(167,229,167,307,TEAL)
text(308,160,'2 original\nchoices',15,MUTED,leading=20)
arrow(291,213,417,213)
text(308,330,'1 edited\nchoice',15,TEAL,leading=20)
arrow(291,347,417,347,TEAL)
line(419,213,419,347,MUTED,2)
arrow(419,280,466,280)
node(470,238,188,84,'Score the\nthree choices',fill='#E7EDF2')
arrow(658,280,702,280)
node(707,238,209,84,'Select and\nexecute',fill='#E0F2EE')
text(44,442,'Ilia keeps its stage tracking and recovery rules.',21,NAVY,True)
text(44,477,'The added networks guide action choice. Predicted value is separate from task success.',14,MUTED)

# 04: Training and the two distinct evaluation panels.
page('Training, comparison, then submission','The development comparison and final evaluation answer different questions.',source='docs/reports/submission-2026/results-snapshot.json')
text(44,131,'20',53,TEAL,True);text(137,150,'demonstrations\nper task',18,NAVY,leading=24)
arrow(360,169,491,169)
text(527,131,'1,000',53,TEAL,True);text(730,150,'training\nupdates',18,NAVY,leading=24)
line(44,231,916,231)
text(44,260,'DEVELOPMENT PILOT',12,MUTED,True)
text(44,287,'36',70,NAVY,True)
text(177,313,'rollouts',25,NAVY,True)
text(44,384,'6 tasks, 3 instances\nOriginal policy versus RT',21,NAVY,leading=31)
line(476,259,476,455)
text(527,260,'FINAL EVALUATION',12,TEAL,True)
text(527,287,'12',70,TEAL,True)
text(660,313,'rollouts',25,NAVY,True)
text(527,384,'3 tasks, public instances 0-3\nRT only, one run per instance',21,NAVY,leading=31)
text(44,476,'The final evaluation has no matched original-policy comparison.',14,MUTED)

# 05: Accurate vector chart built from the frozen pilot measurements.
page('RT did not improve the pilot overall','Mean final goal progress across three development instances per task.',source='docs/reports/submission-2026/results-snapshot.json')
cx,cy,cw,ch=80,165,540,235
for v in [0,.25,.5,.75,1]:
    y=cy+ch-v*ch;line(cx,y,cx+cw,y)
    text(43,y-8,f'{v:.2f}',11,MUTED)
rect(80,127,12,12,NAVY);text(101,123,'Original',14,NAVY)
rect(231,127,12,12,TEAL);text(252,123,'With RT',14,NAVY)
for i,row in enumerate(FACTS['pilot']):
    x=cx+i*90+15
    for j,key in enumerate(['A0_mean_Q','RT_mean_Q']):
        val=row[key];rect(x+j*29,cy+ch-val*ch,24,val*ch,[NAVY,TEAL][j])
    tid=int(row['task_id'])
    text(x-2,411,f'{tid:03d}',13,NAVY,True)
    label=['Radio','Trash','Decor.','Plates','Traps','Toys'][i]
    text(x-9,434,label,11,MUTED)
text(685,129,'MEAN Q',12,MUTED,True)
text(685,162,'0.455',38,NAVY,True);text(685,215,'0.329 with RT',22,TEAL,True)
line(685,261,906,261)
text(685,286,'COMPLETE SUCCESSES',12,MUTED,True)
text(685,322,'5/18',38,NAVY,True);text(685,375,'1/18 with RT',22,TEAL,True)
text(44,476,'Trash gained partial-goal progress, while full-task successes fell.',16,NAVY,True)

# 06: All final runs, including all zeros, shown without selection.
page('Partial progress, no complete successes','All 12 final runs are included. Values show goals satisfied at episode end.',source='results/results.json')
for j in range(4):text(255+j*103,142,f'Public {j}',13,MUTED,True)
for i,(tid,label) in enumerate([(2,'Decorations'),(1,'Trash'),(5,'Mousetraps')]):
    y=190+i*84;text(44,y+7,f'{tid:03d}',13,TEAL,True);text(44,y+29,label,21,NAVY,True)
    vals=[x['official_Q_final'] for x in FINAL['rows'] if x['task_id']==tid]
    for j,v in enumerate(vals):
        x=247+j*103
        fill='#E4EBEF' if v==0 else '#BDE4DF' if v<.6 else '#69B7AF'
        rect(x,y,91,67,fill)
        text(x+14,y+20,f'{100*v:.0f}%',23,NAVY,True)
text(731,170,'0/12',49,ORANGE,True)
text(731,233,'complete\nsuccesses',19,NAVY,leading=25)
text(731,320,'22.4%',35,TEAL,True)
text(731,368,'mean goal\nprogress',18,NAVY,leading=24)
text(44,458,'Q measures progress. Complete success requires all task goals.',17,NAVY,True)
text(44,487,'Self-reported. With 988 unsubmitted instances counted as zero, full-challenge Q = 0.00269048.',11,MUTED)

# 07: Chronological actual frames, selected for explanation rather than scoring.
page('Trash: handling objects did not finish the task','Task 001 / public instance 0',source='docs/submission-2026/task001-failure-analysis/README.md')
for i,(t,caption) in enumerate([(60,'Approach'),(132,'Hold the bin'),(237,'Repeated holding pose')]):
    x=44+i*299
    image(f'task001-public0-t{t:03d}-full.png',x,135,274,191)
    text(x,332,f'{t} s',12,TEAL,True);text(x,357,caption,18,NAVY,True,maxw=278)
text(44,416,'The goal: place three cans inside the bin.',22,NAVY,True)
text(44,455,'Likely bottleneck: insertion and recovery. Video samples do not prove the exact cause.',15,MUTED)
text(44,486,'Simulation time. Head view on the right, wrist views on the left. Final Q: 0.0 in all four instances.',10,MUTED)

# 08: Contrast two instances with their final metric, not the displayed-frame metric.
page('Decorations: outcomes varied by instance','The same policy reached different levels of final goal progress.',source='docs/submission-2026/public3-analysis/README.md')
image('task002-public0-t500-full.png',44,130,417,278)
image('task002-public3-t230-full-from-contact.png',499,130,417,278)
text(44,414,'Public 0',18,NAVY,True);text(250,409,'57% final Q',24,TEAL,True)
text(499,414,'Public 3',18,NAVY,True);text(706,409,'0% final Q',24,ORANGE,True)
text(44,451,'Further object interactions',16,NAVY)
text(499,451,'Close objects block wrist views',16,NAVY)
text(44,486,'Frames at 500 s and 230 s. Occlusion and stalling are observations, not confirmed causes.',11,MUTED)

# 09: Original final recorded views plus final JSON values.
page('Mousetraps: some goals remained unfinished','Final recorded views from the two runs with partial goal progress.',source='results/results.json')
image('task005-public1-last-head.png',44,128,355,300)
image('task005-public3-last-head.png',461,128,355,300)
text(44,441,'Public 1',16,NAVY,True);text(171,435,'50% final Q',24,TEAL,True)
text(461,441,'Public 3',16,NAVY,True);text(588,435,'33% final Q',24,TEAL,True)
text(44,481,'No complete successes in four runs. Final JSON determines scores; images show the task context.',12,MUTED)

# 10: Deliverables and the actual submission state.
page('The submission is complete','The portal accepted the entry on October 11, 2026.',dark=True,source='README.md')
for x,num,label,desc in [(44,'01','Code','Policy and evaluation instructions'),(359,'02','Evidence','12 original videos and result files'),(674,'03','Docker','Embedded model weights')]:
    text(x,158,num,38,'#83D6CF',True)
    text(x,229,label,28,WHITE,True)
    text(x,278,desc,18,'#CAD6DE',maxw=243,leading=27)
line(44,372,916,372,'#385060')
text(44,404,'Organizer verification is pending.',27,WHITE,True)
text(44,458,'L40S deployment checks passed. Actual 24GB hardware remains untested.',15,'#CAD6DE')

# 11: Explicitly future work, not unmeasured results.
page('Next research priorities','Planned work, separate from the submitted policy.',source='docs/submission-2026/task001-failure-analysis/README.md')
image('task005-public0-t100-head.png',582,135,333,333)
priorities=[('01','Placement','Measure whether the object reaches\nand stays in the target region.'),('02','Recovery','Detect repeated poses and learn\nhow to release, retreat, and retry.'),('03','Fair comparison','Compare original and edited actions\non the same instances and settings.')]
for i,(n,title,body) in enumerate(priorities):
    y=142+i*113
    text(44,y,n,27,TEAL,True)
    text(111,y,title,23,NAVY,True)
    text(111,y+38,body,16,MUTED,leading=23)
text(582,477,'An existing rollout, not a new experiment.',10,MUTED)

# 12: Clickable entry points; detailed technical discussion remains in the paper.
page('Explore the evidence','Public files preserve the experiment and support reproduction.')
links=[('Project overview',GITHUB,'Method, results, and current status'),('Evaluation guide',BASE+'EVALUATION-README.md','Docker launch and exact evaluation setup'),('Original rollout videos',BASE+'VIDEOS.md','All 12 recordings, including every failure'),('Research paper',BASE+'docs/reports/submission-2026/research-paper-pilot6-submission12.pdf','Technical details, analysis, and data appendix')]
for i,(label,url,detail) in enumerate(links):
    y=137+i*75
    link(44,y,label,url,23,TEAL)
    text(410,y+5,detail,16,MUTED,maxw=480)
    if i<3:line(44,y+57,916,y+57)
text(44,455,'Data: frozen pilot and final evaluation records. Images: actual demonstrations and rollout frames.',12,MUTED)
text(44,480,'Illustrative frame selections explain behavior; the original videos and metric files stay unchanged.',11,MUTED)
c.save()
qa = ROOT / 'tmp/pdfs'
qa.mkdir(parents=True, exist_ok=True)
(qa/'presentation-layout.json').write_text(json.dumps({'pages':PAGE,'size_pt':[W,H],'text':TEXT_BOXES},indent=2))
print(f'Wrote {PDF} ({PAGE} slides)')
