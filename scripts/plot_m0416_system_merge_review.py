"""Observed JWST morphology for proposed catalog groupings; no catalog edits or lens predictions."""
import sys,json,argparse,textwrap
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from lenscluster import artifacts
from lenscluster.image_catalog import load_image_catalog
from lenscluster.cutouts import imaging_variants,load_cluster_imaging,rgb_reference_band,rgb_cutout,cutout_window
from lenscluster.plot_style import PAPER_STYLE

# Explicit proposed image associations from the supplied review notes, not inferred by label suffix.
GROUPS=[
 ('G1',['K6','K7'], [('a',64.041449,-24.061972,6.,['K6.a','K7.a']),('b',64.043036,-24.063004,5.4,['K6.b','K7.b']),('c',64.047616,-24.068891,2.9,['K6.c','K7.c'])], 'Bergamini system 5: K7 = 5.1; K6 A–E = 5.2–5.6. Does K7 sit in the same light envelope beyond K6A in all three images?'),
 ('G2',['K10','K76'], [('a',64.030911,-24.067244,2.5,['K10.a','K76.a']),('b',64.035208,-24.070967,2.5,['K10.b','K76.b']),('c',64.041981,-24.075871,2.7,['K10.c','K76.c'])], 'Bergamini system 15: K10 = 15.1; K76 A/B = 15.2/15.4. Compare the order of the three knots and the surrounding light in a, b and c.'),
 ('G3',['K36','K37','K40'], [('a',64.040068,-24.066694,2.5,['K36.a','K37.a','K40.b']),('b',64.045391,-24.072537,2.9,['K36.b','K37.b','K40.a'])], 'Bergamini system 9: K36/K37/K40 = 9.1/9.2/9.3. K40 letters are reversed. Is K40 a knot within the same galaxy, or a separate companion?'),
 ('G4',['K69','K70'], [('a',64.043557,-24.058977,2.5,['K69.a','K70.b']),('b',64.048071,-24.062251,2.7,['K69.b','K70.c']),('c',64.050900,-24.066497,2.5,['K69.c','K70.a'])], 'Bergamini system 2: K69/K70 = 2.1/2.2. Look for two clumps inside one envelope. The nearby K72/K74 objects are not part of this proposed group.'),
 ('U1',['K33','K77'], [('north',64.034259,-24.066750,3.6,['K33.a','K33.b','K33.c','K33.e','K77.a','K77.b','K77.c']),('K33.f',64.035245,-24.067941,2.5,['K33.f']),('K33.d',64.046124,-24.076814,2.5,['K33.d'])], 'Bergamini system 14. Check K77.c near K33.c versus the published 14.2c position near K33.e. Look for a second clump near K33.f and K33.d; these are not guaranteed to contain every knot.'),
 ('U2',['K89','K90','K91'], [('a',64.027976,-24.070849,3.,['K89.a','K90.a','K91.a']),('b',64.033172,-24.074152,2.8,['K89.b','K90.b','K91.b']),('c',64.040705,-24.080165,2.8,['K89.c','K90.c','K91.c'])], 'No Bergamini counterparts in the local cross-match. Do the three clumps share an envelope in every image, or are they separate galaxies? Similar redshift alone does not decide.'),
 ('U3',['K72','K73','K74'], [('a',64.050949,-24.066189,4.,['K72.a','K74.a']),('K72 b',64.047239,-24.061214,2.5,['K72.b']),('K74 b',64.049292,-24.063362,2.5,['K74.b']),('K72 c',64.044519,-24.059329,2.5,['K72.c']),('K73 a',64.041564,-24.062654,2.5,['K73.a']),('K73 b',64.040174,-24.061894,2.5,['K73.b'])], 'Bergamini system 210: K74 = 210.1; K72 A/B = 210.2/210.3; K73 = 210.4. These panels are spatially separated. Is there connecting light between K74.a and K72B.a? A shared published prefix does not establish one galaxy.'),
 ('S1',['C2','K29'], [('C2 a / K29 c',64.037226,-24.082406,4.5,['C2.a','K29.c']),('C2 b / K29 a',64.024872,-24.073498,4.5,['C2.b','K29.a']),('C2 c / K29 b',64.030753,-24.079548,4.5,['C2.c','K29.b'])], 'Separation control: compare the distinct light envelopes. The catalog redshifts differ by about 590 km/s; proximity alone is insufficient to group these sources.')]
BERG={
'U1':[('14.1a',64.0345568,-24.0669778),('14.1b',64.0342526,-24.0665069),('14.1c',64.0340651,-24.0664611),('14.1d',64.0340318,-24.0669236),('14.1e',64.0352359,-24.0679408),('14.1f',64.0461244,-24.0768110),('14.2a',64.0345318,-24.0668819),('14.2b',64.0343739,-24.0665700),('14.2c',64.0340651,-24.0664611)],
'U3':[('210.1a',64.0507568,-24.0658475),('210.1b',64.0493009,-24.0633666),('210.2b',64.0472788,-24.0612661),('210.2c',64.0444868,-24.0593175),('210.3a',64.0511430,-24.0665341),('210.3b',64.0471913,-24.0611608),('210.3c',64.0445547,-24.0593450),('210.4b',64.0415597,-24.0626525),('210.4c',64.0401697,-24.0618958)]}

def labels(ax,items,fig):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 occupied=[t.get_window_extent(renderer).expanded(1.05,1.1) for t in ax.texts]
 markers=[Bbox.from_bounds(*(ax.transData.transform(p)-5),10,10) for _,p,_ in items]
 for label,point,color in items:
  t=ax.annotate(label,point,xytext=(6,6),textcoords='offset points',fontsize=7,color=color,zorder=8,arrowprops=dict(arrowstyle='-',lw=.35,color=color,alpha=.75))
  candidates=[]
  for radius in (7,13,20,28):
   for dx,dy in [(1,1),(-1,1),(1,-1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0)]:
    t.set_position((radius*dx,radius*dy));t.set_ha('left' if dx>0 else 'right' if dx<0 else 'center');t.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')
    box=t.get_window_extent(renderer).expanded(1.05,1.1)
    score=1000*(not(ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1)))+100*sum(box.overlaps(o)for o in occupied+markers)+radius
    candidates.append((score,t.get_position(),t.get_ha(),t.get_va()))
  _,pos,ha,va=min(candidates,key=lambda x:x[0]);t.set_position(pos);t.set_ha(ha);t.set_va(va);occupied.append(t.get_window_extent(renderer).expanded(1.05,1.1))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=Path('output/m0416_system_merge_review'));args=parser.parse_args()
 table=load_image_catalog(Path('catalogs/m0416/canucs2025/images.yaml'),sigma_inst_arcsec=1.)
 config=artifacts.load_config(Path('results/nov5a_nosigmainst/m0416_canucs2025/w500s250_allblocks/final/artifacts'))
 view=next(v for v in imaging_variants(config,overview_only=True)if v.figure_suffix=='_jwst')
 images,bands,display,_=load_cluster_imaging(view,config.model,config.paths.run_name);ref=images[rgb_reference_band(bands)]
 plt.style.use(Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
 out=args.output_dir;out.mkdir(parents=True,exist_ok=True);records=[]
 with PdfPages(out/'m0416_system_merge_review.pdf')as combined:
  for gid,systems,panels,note in GROUPS:
   colors=dict(zip(systems,['#55B9FF','#FFB44C','#7AE3AB']));markers=dict(zip(systems,['x','+','D']))
   n=len(panels);fig,axes=plt.subplots(2,n,figsize=(3.35*n,8.25),squeeze=False)
   fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.20,wspace=.01,hspace=.01)
   z={s:float(table.loc[table.system_id==s,'catalog_z'].iloc[0]) for s in systems}
   for col,(name,ra,dec,size,associations)in enumerate(panels):
    # Ensure both members fit even when the supplied control centers sit on one galaxy.
    selected=table.loc[(table.system_id+'.'+table.image_id).isin(associations)]
    assert len(selected)>0,(gid,name)
    center=SkyCoord(ra*u.deg,dec*u.deg);win=cutout_window(ref,center,size)
    if win is None:raise ValueError(f'No coverage: {gid} {name}')
    rgb=rgb_cutout(images,bands,display,center,size)
    def pixels(ra,dec):
     x,y=ref.wcs.world_to_pixel_values(ra,dec);return np.array([x-win.x0,y-win.y0])
    for row in range(2):
     ax=axes[row,col];ax.imshow(rgb,origin='lower',interpolation='nearest');ax.set(xlim=(-.5,rgb.shape[1]-.5),ylim=(-.5,rgb.shape[0]-.5),xticks=[],yticks=[]);ax.grid(False)
     ax.text(.025,.97,f'{gid}  {name}'+('  (unmarked)'if row else ''),transform=ax.transAxes,va='top',fontsize=9,color='white',bbox=dict(facecolor='black',alpha=.55,edgecolor='none',pad=2))
     bar=.5/ref.pixel_scale_arcsec;x0=.07*rgb.shape[1];y0=.065*rgb.shape[0]
     ax.plot([x0,x0+bar],[y0,y0],color='white',lw=1.2);ax.text(x0+bar/2,y0+.04*rgb.shape[0],r'$0.5^{\prime\prime}$',ha='center',color='white',fontsize=8)
     ax.text(.97,.035,f'{size:g}″ across',transform=ax.transAxes,ha='right',color='white',fontsize=7)
    ax=axes[0,col];items=[]
    for _,r in selected.iterrows():
     point=pixels(r.ra,r.dec);assert np.all(point>=-.5)and point[0]<rgb.shape[1]and point[1]<rgb.shape[0],(gid,r.image_label,point)
     ax.plot(*point,marker=markers[r.system_id],ms=7,mew=.9,mfc='none',ls='none',color=colors[r.system_id]);items.append((r.image_label,point,colors[r.system_id]))
    literature={}
    for label,bra,bdec in BERG.get(gid,[]):
     literature.setdefault((bra,bdec),[]).append(label)
    literature_items=[]
    for (bra,bdec),names in literature.items():
     label=' / '.join(names)
     point=pixels(bra,bdec)
     if np.all(point>=0)and point[0]<rgb.shape[1]and point[1]<rgb.shape[0]:
      ax.plot(*point,'o',mfc='none',mec='white',ms=9,mew=.65)
      # Literature labels are on the otherwise clean copy to keep knot IDs readable.
      axes[1,col].plot(*point,'o',mfc='none',mec='white',ms=7,mew=.65)
      literature_items.append((label,point,'white'))
    if gid in BERG:axes[1,col].texts[0].set_text(f'{gid}  {name}  (Bergamini positions)')
    labels(ax,items,fig)
    labels(axes[1,col],literature_items,fig)
    coords=SkyCoord(selected.ra.to_numpy()*u.deg,selected.dec.to_numpy()*u.deg);sep=coords[:,None].separation(coords[None,:]).arcsec
    records.append(dict(group=gid,panel=name,center_deg=[ra,dec],size_arcsec=size,labels=selected.image_label.tolist(),separations_arcsec=sep.tolist()))
   handles=[Line2D([],[],color=colors[s],marker=markers[s],ls='none',mfc='none',label=f'{s}: z = {z[s]:g}')for s in systems]
   if gid in BERG:handles.append(Line2D([],[],color='0.3',marker='o',mfc='none',ls='none',label='Bergamini (supplied shifted coordinates)'))
   fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.145),ncol=len(handles),frameon=False,fontsize=9)
   maxdv=299792.458*(max(z.values())-min(z.values()))/(1+np.mean(list(z.values())))
   caption=' + '.join(systems)+f'   |   Catalog redshift span: {maxdv:.0f} km/s.  '+note
   fig.text(.025,.122,textwrap.fill(caption,width=int(45*n)),va='top',fontsize=9)
   footer='RGB: B = F115W + F150W; G = F150W + F200W; R = F444W. Same stretch throughout. Proposed associations, not confirmed merges. Compare light envelopes and repeated knots; colour alone does not establish a shared source.'
   fig.text(.025,.015,textwrap.fill(footer,width=int(51*n)),fontsize=7.5,va='bottom')
   path=out/f'{gid}_{"_".join(systems)}.pdf'
   fig.savefig(path,dpi=300);combined.savefig(fig,dpi=300);plt.close(fig);print(path,flush=True)
 (out/'panel_positions.json').write_text(json.dumps(dict(catalog='catalogs/m0416/canucs2025/images.yaml',bands=list(bands),panels=records),indent=2))
if __name__=='__main__':main()
