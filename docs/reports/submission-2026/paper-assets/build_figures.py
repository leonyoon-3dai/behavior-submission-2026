"""Rebuild exact-data manuscript figures; never modifies rollout evidence."""
from pathlib import Path
import json, shutil, hashlib, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=Path(__file__).resolve().parent
REPORT=HERE.parent
ROOT=REPORT.parents[2]
# Optional import from an already verified frame bundle; committed PNGs suffice.
FRAMES=Path(os.environ['PAPER_FRAME_SOURCE']) if os.environ.get('PAPER_FRAME_SOURCE') else None
INK='#183047'; TEAL='#078B83'; BLUE='#467CA9'; GRAY='#647586'; LIGHT='#EAF2F5'; ORANGE='#D08744'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':INK,'pdf.fonttype':42})

frames=[
'task002-public0-t300-head.png','task001-public0-t184-head.png','task005-public1-t100-head.png',
'task001-public0-t060-full.png','task001-public0-t132-full.png','task001-public0-t184-full.png','task001-public0-t237-full.png',
'task002-public0-t050-head.png','task002-public0-t500-head.png','task005-public1-last-head.png','task005-public3-last-head.png',
]
if FRAMES is not None and FRAMES.exists():
    source=json.loads((FRAMES/'provenance.json').read_text())
    items=[]
    for name in frames:
        shutil.copyfile(FRAMES/name,HERE/name)
        item=next(x.copy() for x in source['items'] if x['file']==name)
        item['paper_file']='paper-assets/'+name
        items.append(item)
    (HERE/'provenance.json').write_text(json.dumps({'source':'Original final rollout videos. Crops only; no content retouching.','items':items},indent=2)+'\n')

def box(ax,x,y,w,h,label,color=LIGHT,edge=None,size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',facecolor=color,edgecolor=edge or color,linewidth=1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,linespacing=1.5)
def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=color,lw=1.4,connectionstyle='arc3'))

fig,ax=plt.subplots(figsize=(8.4,3.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.015,.95,'AT EACH REPLAN',weight='bold',color=TEAL,fontsize=10)
box(ax,.015,.47,.18,.32,'Observe\n3 RGB views\n+ robot state')
box(ax,.265,.47,.19,.32,'Frozen Ilia\n2 proposed\naction sequences')
box(ax,.53,.47,.18,.32,'Small edit\n+1 alternative\n(3 total)')
box(ax,.795,.47,.19,.32,'Score & select\nTwin target critic\nbest of 3')
for a,b in [((.2,.63),(.25,.63)),((.46,.63),(.52,.63)),((.72,.63),(.785,.63))]:arrow(ax,a,b)
box(ax,.795,.04,.19,.27,'Execute\nIlia rules\n15 actions',color='#DDEFE9')
arrow(ax,(.89,.46),(.89,.33))
ax.text(.02,.24,'TRAINED FOR EACH TASK',color=TEAL,fontsize=9,weight='bold')
ax.text(.02,.11,'20 demos  /  1,000 auxiliary updates\nOriginal Ilia weights stay unchanged.',fontsize=10,linespacing=1.55)
ax.text(.53,.19,'4-action carry\nconditions next chunk',ha='center',va='center',fontsize=9,color=GRAY)
arrow(ax,(.79,.13),(.68,.13));arrow(ax,(.42,.14),(.36,.46))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1);fig.savefig(HERE/'method-overview.pdf');plt.close(fig)

snap=json.loads((REPORT/'results-snapshot.json').read_text())
tasks=['0','1','2','3','5','7']; labels=['000  Radio','001  Trash','002  Decorations','003  Plates & food','005  Mousetraps','007  Toys']
a=np.array([snap['pilot6']['tasks'][t]['conditions']['A0-original']['mean_Q'] for t in tasks]);r=np.array([snap['pilot6']['tasks'][t]['conditions']['RT-EXPO']['mean_Q'] for t in tasks]);y=np.arange(6)
fig,ax=plt.subplots(figsize=(8.4,3.7));ax.barh(y-.16,a,.27,color=BLUE,label='Original Ilia (A0)');ax.barh(y+.16,r,.27,color=TEAL,label='Ilia + RT editing');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,.83);ax.set_xticks([0,.2,.4,.6,.8]);ax.set_xlabel('Mean final goal progress Q  (3 rollouts per bar)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0)
for i,(aa,rr) in enumerate(zip(a,r)):
 ax.text(aa+.015,i-.16,f'{aa:.3f}',va='center',fontsize=9);ax.text(rr+.015,i+.16,f'{rr:.3f}',va='center',fontsize=9)
ax.legend(loc='upper center',bbox_to_anchor=(.56,1.15),ncol=2,frameon=False,fontsize=10);fig.tight_layout();fig.savefig(HERE/'pilot-results.pdf');plt.close(fig)

rows=snap['submission12']['rows']
q=np.array([[next(r['Q_final'] for r in rows if r['task_id']==task and r['public_index']==idx) for idx in range(4)] for task in [2,1,5]])
fig,ax=plt.subplots(figsize=(8.4,2.8));cmap=matplotlib.colors.LinearSegmentedColormap.from_list('goal',['#EEF2F5','#61B2A9','#067F76']);ax.imshow(q,cmap=cmap,vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(3),['002  Decorations','001  Trash','005  Mousetraps']);ax.set_xticks(range(4),['Public 0','Public 1','Public 2','Public 3']);ax.xaxis.tick_top();ax.tick_params(length=0,pad=9);ax.spines[:].set_visible(False)
for i in range(3):
 for j in range(4):ax.text(j,i,f'{q[i,j]:.3f}',ha='center',va='center',fontsize=16,color='white' if q[i,j]>.55 else INK,weight='bold')
ax.set_xticks(np.arange(-.5,4,.1) if False else np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True);ax.grid(which='minor',color='white',linewidth=8);ax.tick_params(which='minor',length=0);fig.tight_layout();fig.savefig(HERE/'final-results.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.4,2.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.01,.55,.28,.30,'Human demonstrations\nimages + state + actions')
box(ax,.36,.55,.28,.30,'Learning coordinates\nnormalize + align frames')
box(ax,.71,.55,.28,.30,'Small auxiliary models\ncritic + edit + encoder')
arrow(ax,(.30,.70),(.35,.70));arrow(ax,(.65,.70),(.70,.70))
box(ax,.01,.06,.28,.27,'Ilia candidate\n30 steps × 32 padded dims',color='#E9EEF9')
box(ax,.36,.06,.28,.27,'Physical / critic bridge\n23 commanded dims')
box(ax,.71,.06,.28,.27,'Robot execution\n15 steps + 4 carry',color='#DDEFE9')
arrow(ax,(.30,.19),(.35,.19));arrow(ax,(.65,.19),(.70,.19));arrow(ax,(.85,.54),(.85,.36))
ax.text(.015,.94,'DATA FOR TRAINING',fontsize=10,weight='bold',color=TEAL);ax.text(.015,.41,'DATA DURING INFERENCE',fontsize=10,weight='bold',color=TEAL)
fig.subplots_adjust(left=0,right=1,bottom=0,top=1);fig.savefig(HERE/'representation-overview.pdf');plt.close(fig)

print('Wrote four vector figures;',len(frames),'rollout frame assets referenced.')
