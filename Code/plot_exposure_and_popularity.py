from pathlib import Path
import numpy as np,fitz, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
project=Path(__file__).resolve().parents[1]
root=project/'Figures/Results'
root.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True})
names=['Global quantity','Category quantity','ItemKNN','BPR-MF','NCF']
p=root/'category_availability_map.pdf'
d=fitz.open(project/'Inputs/category_availability_map.pdf');page=d[0]
images=page.get_images()
boxes=[v['items'][0][1] for v in page.get_drawings() if v['fill']==(1.,1.,1.) and v['items'][0][0]=='re' and 170<v['rect'].width<180]
assert len(boxes)==len(images)==5
fig,aa=plt.subplots(2,3,figsize=(9.4,5.35));fig.subplots_adjust(left=.092,right=.975,bottom=.14,top=.93,wspace=.30,hspace=.43)
for a,model,box,im in zip(aa.flat,names,boxes,images):
 x,mask,*_=im;r=page.get_image_rects(x)[0]
 pix=fitz.Pixmap(d,x)
 if pix.colorspace.n!=3:pix=fitz.Pixmap(fitz.csRGB,pix)
 arr=np.array(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
 maskpix=fitz.Pixmap(d,mask)
 alpha=np.asarray(Image.frombytes('L',(maskpix.width,maskpix.height),maskpix.samples))/255
 arr=np.dstack((arr/255,np.clip(alpha**.53,0,1)))
 ext=((r.x0-box.x0)/box.width,(r.x1-box.x0)/box.width,(box.y1-r.y1)/box.height,(box.y1-r.y0)/box.height)
 a.imshow(arr,extent=ext,origin='upper',aspect='auto',interpolation='nearest',zorder=3)
 tick=(np.array([0,33.16949,75.81596,118.46243,161.10891])/173.9468)
 labels=['0','0.01','0.1','1','10']
 a.set(xlim=(0,1),ylim=(0,1),xticks=tick,yticks=tick,xticklabels=labels,yticklabels=labels)
 a.plot([0,1],[0,1],ls='--',lw=1,color='#87939a',zorder=2)
 a.grid(color='#eceff0',lw=.6)
 a.set_title(model,pad=8,fontsize=12)
 a.tick_params(length=3,pad=3,labelsize=10)
aa[1,2].axis('off')
aa[1,2].legend(handles=[Line2D([],[],marker='o',ls='',color='#487eaa',label='Below reference'),Line2D([],[],marker='o',ls='',color='#cf8541',label='Above reference'),Line2D([],[],ls='--',color='#87939a',label='Equal shares')],loc='upper left',frameon=False,borderaxespad=0,fontsize=11)
aa[1,2].text(0,.23,'Linear below 0.01%;\nlogarithmic above.',transform=aa[1,2].transAxes,fontsize=10)
fig.supxlabel('Candidate share (%)',y=.025,fontsize=12)
fig.supylabel('Recommendation share (%)',x=.015,fontsize=12)
d.close()
fig.savefig(p);plt.close(fig)
# Add the observed training-purchase mix to the existing two-panel comparison.
data=np.array([[25.779114,26.321685,47.899201],[100,0,0],[28.93,16.94,54.12],[12.33,13.10,74.58],[31.74,30.67,37.59],[16.99,23.26,59.76]])
# Printed model values are rounded; normalize only the rendering of the stacks.
data=data/data.sum(axis=1)[:,None]*100
models=['Training purchase mix',*names]
fig,(a,b)=plt.subplots(1,2,figsize=(9.2,3.55),gridspec_kw={'width_ratios':[2.25,1]},layout='constrained')
left=np.zeros(len(models));ys=np.arange(len(models))
for values,label,color in zip(data.T,['Head','Mid','Tail'],['#264b73','#d99b35','#169a8a']):
 a.barh(ys,values,left=left,height=.62,label=label,color=color)
 left+=values
a.set(yticks=ys,yticklabels=models,ylim=(len(models)-.5,-.5),xlim=(0,100),xlabel='Share (%)')
a.legend(frameon=False,ncols=3,loc='lower center',bbox_to_anchor=(.5,1.02))
b.barh(ys[1:],[.10,18.62,54.77,41.21,4.88],height=.62,color='#264b73')
for i,v in enumerate([.10,18.62,54.77,41.21,4.88],1):b.text(v+1,i,f'{v:.2f}',va='center',fontsize=10)
b.text(0,0,'Not applicable',va='center',fontsize=10,color='#666666')
b.set(yticks=[],ylim=(len(models)-.5,-.5),xlim=(0,65),xlabel='Catalog coverage (%)')
for ax in (a,b):ax.grid(axis='x',color='#edf0f1')
fig.savefig(root/'chronological20_popularity.pdf');plt.close(fig)
