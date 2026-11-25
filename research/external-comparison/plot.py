"""Render the checked-in aggregate CSV as an accessible GitHub SVG."""
import csv
import math
from pathlib import Path
from html import escape

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'docs' / 'assets' / 'external-benchmark.svg'
with (HERE / 'metrics.csv').open(newline='') as handle:
    data = {(r['method'], r['slice']): r for r in csv.DictReader(handle) if r['seed'] == '42'}

svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 688" role="img" aria-labelledby="t d">',
       '<title id="t">Order-fault detection across eight methods</title>',
       '<desc id="d">Raw and milestone-input models compared on 192 synthetic pairs. Milestone-focused order head, BERT-Mini and precedence tie at full recall with no false alerts. Labels give detections, false alerts, and throughput.</desc>']

def label(x,y,string,size=15,fg='#dce7f2',weight='400',align='start'):
    svg.append(f'<text x="{x}" y="{y}" fill="{fg}" font-size="{size}" font-weight="{weight}" text-anchor="{align}">{escape(str(string))}</text>')

def rect(x,y,w,h,fill,corner=0):
    svg.append(f'<rect x="{x}" y="{y}" width="{w:.3f}" height="{h}" fill="{fill}" rx="{corner}"/>')

svg.append('<g font-family="system-ui,sans-serif">')
rect(0,0,1100,688,'#0c1725',18)
label(36,47,'EQUIP TRIAGE   /   SYNTHETIC MODEL COMPARISON',16,'#75e3c5','700')
label(36,94,'Retain the milestones, not every poll',33,'#f7fbff','750')
label(36,126,'192 matched test pairs · seed 42 · validation-normal threshold targeting ≤1% false alerts',16,'#a9bdce')
rect(36,151,12,12,'#75e3c5',3)
label(56,162,'Familiar fault (96)',14)
rect(219,151,12,12,'#83abff',3)
label(239,162,'Withheld fault (96)',14)
label(233,200,'DETECTIONS / 96 PER FAULT SLICE',13,'#b5cce1','700')
label(727,200,'FALSE ALERTS / 192',13,'#b5cce1','700','middle')
label(1064,200,'TRACES / CPU SECOND',13,'#b5cce1','700','end')
models = [('mlm','Custom MLM'),('order','Custom order head'),('tiny_pretrained','BERT-Tiny'),('mini_pretrained','BERT-Mini'),
          ('smol','SmolLM2 135M · 4-shot'),('milestone_order','Custom head + milestones'),('mini_milestones','BERT-Mini + milestones'),('precedence','Learned precedence')]
for idx,(key,name) in enumerate(models):
    base = data[key,'all']
    seen,held=(int(data[key,s]['tp']) for s in ('seen_scan','held_release'))
    y=243+idx*49+(14 if idx>=5 else 0)
    if idx==5:
        rect(36,y-40,1028,1,'#456075')
        label(36,y-22,'MILESTONE INPUTS OR RULES',11,'#75e3c5','700')
    if idx%2==0:
        rect(31,y-17,1038,43,'#122538',6)
    label(36,y+8,name,14,'#dce7f2','650' if idx>=5 else '400')
    for v,dy,c in ((seen,0,'#75e3c5'),(held,13,'#83abff')):
        rect(233,y-7+dy,350,7,'#25384b',2)
        if v:
            rect(233,y-7+dy,350*v/96,7,c,2)
        label(599,y+dy+1,v,12,c)
    label(727,y+8,base['fp'],16,'#f7fbff','650','middle')
    speed=float(base['traces_per_cpu_s'])
    rect(835,y-2,110,6,'#25384b',2)
    rect(835,y-2,110*max(.015,math.log10(speed)/6),6,'#c4a5ff',2)
    label(1064,y+8,f'{speed:,.0f}' if speed>=100 else f'{speed:.1f}',15,'#efe4ff','500','end')
label(36,662,'CPU bars use a log scale. BERT models are fine-tuned; the LLM uses four examples and no weight updates.',13,'#a9bdce')
svg.append('</g></svg>')
OUT.write_text('\n'.join(svg)+'\n', encoding='utf-8')
