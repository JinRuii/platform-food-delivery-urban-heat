#!/usr/bin/env python3
"""Reproduce the selected-date hourly precipitation figure.

Inputs: the bundled 120-row CSV, without further downloads.
Outputs: 600 dpi PNG, editable SVG, and outlined SVG.
Usage: python plot_precipitation.py [--font-path /path/to/times.ttf]
The half-hour window range is an allocation bound, not a confidence interval.
"""
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import csv, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import font_manager
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Reproduce the precipitation figure in Times New Roman.")
parser.add_argument("--data", type=Path, default=root / "data" / "UK_5Cities_PeakDates_2024_Hourly_Precipitation.csv")
parser.add_argument("--output-dir", type=Path, default=root / "figures")
parser.add_argument("--font-path", type=Path, help="Optional path to a licensed Times New Roman TTF/OTF font.")
args = parser.parse_args()
if args.font_path:
    if not args.font_path.is_file():
        parser.error("--font-path does not name an existing file")
    font_manager.fontManager.addfont(str(args.font_path))
    actual_family = font_manager.FontProperties(fname=str(args.font_path)).get_name()
    if actual_family != "Times New Roman":
        parser.error(f"Expected Times New Roman, but supplied font is {actual_family!r}")
try:
    font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
except ValueError:
    parser.error("Times New Roman was not found; install it or pass --font-path /path/to/times.ttf")
if not args.data.is_file():
    parser.error(f"Input CSV not found: {args.data}")
args.output_dir.mkdir(parents=True, exist_ok=True)
source=args.data
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
cities=['London','Birmingham','Manchester','Leeds','Glasgow']
data={c:sorted([r for r in rows if r['city']==c], key=lambda r:r['hour_start_utc']) for c in cities}
assert len(rows)==120 and all(len(data[c])==24 for c in cities)
matrix=np.array([[float(r['precipitation_mm']) for r in data[c]] for c in cities])

def bounds(records,start,end):
 lo=Decimal('0');hi=Decimal('0')
 for r in records:
  d=datetime.fromisoformat(r['hour_start_local']);a=d.hour+d.minute/60;b=a+1
  p=Decimal(r['precipitation_mm'])
  if a>=start and b<=end:lo+=p
  if min(b,end)>max(a,start):hi+=p
 return float(lo),float(hi)

lunch='#E74C3C';dinner='#9B59B6';ink='#222222';rain='#3498DB';gray='#777777'
plt.rcParams.update({
 'font.family':'serif','font.serif':['Times New Roman'],
 'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'xtick.labelsize':7,
 'ytick.labelsize':7.5,'axes.linewidth':.6,'xtick.major.width':.6,
 'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'text.color':ink,'axes.labelcolor':ink,'xtick.color':ink,'ytick.color':ink,
 'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none',
})
fig=plt.figure(figsize=(7.2,4.3))
ax=fig.add_axes([.17,.39,.515,.47])
bx=fig.add_axes([.755,.39,.215,.47])
fig.text(.037,.925,'a',fontsize=11,fontweight='bold')
fig.text(.17,.925,'Hourly precipitation',fontsize=8.5)
fig.text(.715,.925,'b',fontsize=11,fontweight='bold')
fig.text(.755,.925,'Window totals',fontsize=8.5)

cmap=LinearSegmentedColormap.from_list('precip',['#FAFAFA','#A9D4EF','#3498DB'])
norm=Normalize(vmin=0,vmax=.2)
mesh=ax.pcolormesh(np.arange(25),np.arange(6),matrix,cmap=cmap,norm=norm,edgecolors='#E4E4E4',linewidth=.35,rasterized=False)
ax.set_xlim(0,24);ax.set_ylim(5,0)
ax.set_xticks([0,4,8,12,16,20,24]);ax.set_xticklabels(['00','04','08','12','16','20','24'])
ax.set_xlabel('Local time (BST)',labelpad=4)
ax.set_yticks(np.arange(5)+.5)
ax.set_yticklabels([f"{c}\n{data[c][0]['local_date']}" for c in cities],linespacing=1.4)
ax.tick_params(axis='y',length=0,pad=5)
for sp in ax.spines.values():sp.set_visible(False)
for a,b,color,label in [(11.5,13.5,lunch,'Lunch'),(17.5,19.5,dinner,'Dinner')]:
 ax.add_patch(Rectangle((a,0),b-a,5,fill=False,edgecolor=color,lw=.8,zorder=3))
 ax.plot([a,b],[1.025,1.025],transform=ax.get_xaxis_transform(),color=color,lw=1.5,clip_on=False)
 ax.text((a+b)/2,1.06,label,color=color,ha='center',va='bottom',fontsize=7,transform=ax.get_xaxis_transform())
for i in range(5):
 for h in range(24):
  v=matrix[i,h]
  if v>0:
   ax.text(h+.5,i+.5,f'{v:.1f}',ha='center',va='center',fontsize=6.3,
           color='white' if v>.15 else ink,zorder=5,
           bbox={'facecolor':cmap(norm(v)),'edgecolor':'none','pad':.05})

# Three window totals per city. A range denotes an unknown sub-hour allocation,
# not a confidence interval and not a uniformly interpolated point estimate.
styles=[('All day',0,24,ink,'o',-.17),('Lunch peak',11.5,13.5,lunch,'s',0),('Dinner peak',17.5,19.5,dinner,'^',.17)]
for i,c in enumerate(cities):
 for label,a,b,color,marker,offset in styles:
  lo,hi=bounds(data[c],a,b);y=i+.5+offset
  if lo==hi:
   bx.plot(lo,y,marker=marker,markersize=3.7,color=color,markeredgewidth=.55,linestyle='none',clip_on=False,zorder=4)
  else:
   bx.hlines(y,lo,hi,color=color,lw=1.1,zorder=4)
   bx.vlines([lo,hi],y-.055,y+.055,color=color,lw=.8,zorder=4)
   bx.text(hi+.014,y,'0\u20130.1',fontsize=6.4,color=color,va='center')
 for value in [.1,.2,.3]:bx.axvline(value,color='#E5E5E5',lw=.45,zorder=0)
bx.set_xlim(-.015,.34);bx.set_ylim(5,0)
bx.set_xticks([0,.1,.2,.3]);bx.set_xticklabels(['0','0.1','0.2','0.3'])
bx.set_yticks([]);bx.set_xlabel('Precipitation (mm)',labelpad=4)
for name in ['left','top','right']:bx.spines[name].set_visible(False)
for y in np.arange(1,5):bx.axhline(y,color='#EBEBEB',lw=.45,zorder=0)

cax=fig.add_axes([.17,.247,.20,.021])
cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[0,.1,.2])
cb.ax.set_xticklabels(['0','0.1','0.2']);cb.outline.set_linewidth(.45)
cb.solids.set_rasterized(False)
cb.ax.tick_params(labelsize=6.5,length=2,pad=1)
cb.set_label('Hourly precipitation (mm)',fontsize=7,labelpad=3)
handles=[Line2D([0],[0],color=color,marker=marker,ls='none',markersize=4,label=label) for label,a,b,color,marker,off in styles]
fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.974,.285),ncol=3,frameon=False,
           fontsize=7,handletextpad=.35,columnspacing=.8,borderaxespad=0)
fig.text(.49,.215,'Lunch 11:30\u201313:30; dinner 17:30\u201319:30',fontsize=7,color='#444444')
fig.text(.037,.142,'ERA5 via Open-Meteo; nearest city-centre grid cells (~25 km). All times are UK local time.',fontsize=6.6,color='#555555')
fig.text(.037,.108,'The lunch range reflects an unresolved half-hour boundary; no within-hour rainfall distribution is assumed.',fontsize=6.6,color='#555555')
fig.text(.037,.074,'These selected peak dates do not cover every heat-exposure city-day in the full analysis.',fontsize=6.6,color='#555555')

stem = "Hourly_Precipitation_Delivery_Peaks_Nature_Times"
outputs = [args.output_dir / (stem + suffix) for suffix in [".png", ".svg", "_outlined.svg"]]
fig.savefig(outputs[0], dpi=600)
# Editable text SVG; all chart elements remain vector objects.
fig.savefig(outputs[1], format="svg")
# Glyphs converted to vector paths for exact appearance without installed fonts.
with plt.rc_context({"svg.fonttype": "path"}):
    fig.savefig(outputs[2], format="svg")
plt.close(fig)
for out in outputs:
    with open(out, "rb") as f:
        os.fsync(f.fileno())
    print(out)
