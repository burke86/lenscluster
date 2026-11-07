"""Temporary published-palette mock corners. Production defaults are untouched.

Run with the lenstronomy interpreter; requires palettable, colorcet,
cmcrameri, seaborn, colorspacious, ggsci, pypalettes and pymupdf. PDF output only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex, to_rgba_array, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import Collection
import numpy as np

from lenscluster.block_corner import render_block_corner, BLOCK_COLORS, STYLE, PAPER_STYLE
from lenscluster.mock_validation.aggregate_plots import PARAMETER_LABELS


def published_palettes():
    import palettable
    import seaborn as sns
    import colorcet as cc
    from cmcrameri import cm
    from colorspacious import cspace_convert
    rows = []
    def add(name, colors, source, accessibility='No blanket colorblind claim'):
        chosen, indices, positions, contrasts = [], [], [], []
        for index, c in enumerate(colors):
            rgb=np.asarray(to_rgba(c)[:3])
            linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            contrast=1.05/(linear @ np.array([.2126,.7152,.0722])+.05)
            if contrast < 3.: continue
            position=cspace_convert(rgb,'sRGB1','CAM02-UCS')
            chroma=float(np.linalg.norm(position[1:]))
            if not (20 <= chroma <= 42 and position[0]>=32 and rgb.max()<=.96):continue
            if any(np.linalg.norm(position-p)<12 for p in positions): continue
            chosen.append(to_hex(c));indices.append(index);positions.append(position);contrasts.append(float(contrast))
            if len(chosen)==5:break
        if len(chosen)<5:return
        if any(r['colors']==chosen for r in rows):return
        rows.append(dict(name=name, colors=chosen, source=source, accessibility=accessibility,
                         published_indices=indices, white_contrast=contrasts))
    import importlib.util
    import runpy
    data_path=Path(importlib.util.find_spec('ggsci').origin).with_name('data.py')
    journal_palettes=runpy.run_path(str(data_path))['PALETTES']
    for family in ['npg','aaas','nejm','lancet','jama','bmj','jco','ucscgb','igv','locuszoom','uchicago','cosmic','frontiers','observable','primer','atlassian','flatui','gephi']:
        for name, colors in journal_palettes[family].items():
            if isinstance(colors,list) and all(isinstance(c,str) for c in colors):
                add('ggsci '+family+' '+name, colors, 'https://nanx.me/ggsci/articles/ggsci.html',
                    'Published journal or data-visualization palette; no blanket CVD guarantee')
    for name in ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3']:
        add('ColorBrewer '+name, getattr(palettable.colorbrewer.qualitative, name+'_'+str(max(int(k.rsplit('_',1)[1]) for k in dir(palettable.colorbrewer.qualitative) if k.startswith(name+'_') and k.rsplit('_',1)[1].isdigit()))).mpl_colors,
            'https://colorbrewer2.org/')
    for name in ['Antique', 'Bold', 'Pastel', 'Prism', 'Safe', 'Vivid']:
        add('CARTO '+name, getattr(palettable.cartocolors.qualitative, name+'_10').mpl_colors,
            'https://carto.com/carto-colors/', 'Published colorblind-friendly scheme' if name=='Safe' else 'No blanket colorblind claim')
    for name in ['deep', 'muted', 'dark', 'pastel', 'colorblind']:
        add('Seaborn '+name, sns.color_palette(name, 10), 'https://seaborn.pydata.org/tutorial/color_palettes.html',
            'Published colorblind palette' if name=='colorblind' else 'No blanket colorblind claim')
    for name in ['TableauLight_10', 'TableauMedium_10', 'ColorBlind_10', 'BlueRed_12', 'GreenOrange_12', 'PurpleGray_12']:
        add('Tableau '+name, getattr(palettable.tableau, name).mpl_colors,
            'https://jiffyclub.github.io/palettable/tableau/',
            'Published colorblind palette' if name=='ColorBlind_10' else 'No blanket colorblind claim')
    add('Petroff 10', matplotlib.color_sequences['petroff10'], 'https://github.com/mpetroff/accessible-color-cycles',
        'Designed to balance accessibility and aesthetics')
    # Published Okabe-Ito colors, with black last; no color modifications.
    add('Okabe Ito', ['#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7','#000000'],
        'https://jfly.uni-koeln.de/color/', 'Published colorblind-friendly scheme')
    for name, colors in [
        ('dark',['#4477BB','#CC4466','#228833','#2255AA','#993366','#006644','#117788','#882288','#BB4488','#775500','#444444']),
        ('bright',['#4477AA','#EE6677','#228833','#CCBB44','#66CCEE','#AA3377','#BBBBBB']),
        ('muted',['#CC6677','#332288','#DDCC77','#117733','#88CCEE','#882255','#44AA99','#999933','#AA4499']),
        ('vibrant',['#EE7733','#0077BB','#33BBEE','#EE3377','#CC3311','#009988','#BBBBBB'])]:
        add('Tol '+name, colors, 'https://sronpersonalpages.nl/~pault/',
            'White-background contrast prioritized; weaker CVD separation' if name=='dark' else 'Published colorblind-safe scheme')
    for name in ['glasbey_light', 'glasbey_dark', 'glasbey_cool', 'glasbey_warm', 'glasbey_bw']:
        add('Colorcet '+name, getattr(cc, name), 'https://colorcet.holoviz.org/user_guide/Categorical.html',
            'Perceptual categorical separation; not a blanket CVD guarantee')
    for name in ['batlowS','batlowKS','batlowWS','glasgowS','hawaiiS','lipariS','naviaS','nuukS','actonS','bamakoS','bilbaoS','budaS','davosS','devonS','imolaS','lajollaS','lapazS','osloS','tokyoS','turkuS']:
        add('Crameri '+name, getattr(cm,name).colors, 'https://www.fabiocrameri.ch/colourmaps/',
            'Published scientific categorical map; inspect selected colors and white-background contrast')
    import csv, ast, pypalettes
    with Path(pypalettes.__file__).with_name('palettes.csv').open() as handle:
        catalog=list(csv.DictReader(handle))
    scientific_sources={'pals':'https://kwstat.github.io/pals/',
                        'Polychrome':'https://cran.r-project.org/package=Polychrome',
                        'dichromat':'https://cran.r-project.org/package=dichromat',
                        'colorBlindness':'https://cran.r-project.org/package=colorBlindness',
                        'ggthemes':'https://jrnold.github.io/ggthemes/'}
    for package,url in scientific_sources.items():
        for r in catalog:
            if r['source']!='The R package: {'+package+'}' or r['kind']!='qualitative':continue
            if package=='ggthemes' and (r['name'].startswith(('Classic','Tableau','excel','hc_')) or r['name'] in ['Traffic','Summer','Winter','Seattle_Grays']):continue
            add(package+' '+r['name'],ast.literal_eval(r['palette']),url,
                'Published scientific/data-visualization palette; selected subset requires CVD inspection')
    assert len(rows)>=36, (len(rows),[r['name'] for r in rows])
    rows=rows[:36]
    assert len(rows)==36 and len({tuple(r['colors']) for r in rows})==36
    return rows


def color_bindings(fig):
    """Capture original artist colors once, including contour transparency."""
    originals = np.array([to_rgba(c)[:3] for c in BLOCK_COLORS.values()])
    bindings = []
    for artist in fig.findobj():
        attributes = []
        if isinstance(artist, (Line2D, Text)):
            attributes = [('color', True)]
        elif isinstance(artist, (Collection, Patch)):
            attributes = [('facecolor', isinstance(artist, Patch)), ('edgecolor', isinstance(artist, Patch))]
        for attr, scalar in attributes:
            rgba = to_rgba_array(getattr(artist,'get_'+attr)()).copy()
            if not len(rgba): continue
            matches = np.all(np.isclose(rgba[:,:3,None], originals.T[None,:,:], atol=1e-9),axis=1)
            if matches.any(): bindings.append((getattr(artist,'set_'+attr),rgba,matches,scalar,getattr(artist,'get_'+attr)))
        if isinstance(artist, Collection) and isinstance(artist.get_cmap(), ListedColormap):
            rgba = to_rgba_array(artist.get_cmap().colors).copy()
            matches = np.all(np.isclose(rgba[:,:3,None], originals.T[None,:,:], atol=1e-9),axis=1)
            if matches.any():
                bindings.append((lambda colors, a=artist: a.set_cmap(ListedColormap(colors)),
                                 rgba,matches,False,lambda a=artist: a.get_cmap().colors))
    return bindings


def geometry_hash(fig):
    h=hashlib.sha256()
    for ax in fig.axes:
        h.update(np.asarray([ax.get_xlim(),ax.get_ylim()]).tobytes())
        for collection in ax.collections:
            for path in collection.get_paths(): h.update(path.vertices.tobytes())
        for line in ax.lines:
            h.update(np.asarray(line.get_xydata()).tobytes())
    return h.hexdigest()


def swatches(rows, output):
    from colorspacious import cspace_convert
    visions=[('Original',None),('Deuteranopia simulation','deuteranomaly'),('Protanopia simulation','protanomaly')]
    with plt.rc_context(fname=STYLE,rc=PAPER_STYLE):
        fig, ax=plt.subplots(figsize=(15,17))
        ax.set_xlim(0,16);ax.set_ylim(-1,37);ax.axis('off')
        for j,(label,cvd) in enumerate(visions):ax.text(5.4+j*3.5,36.5,label,fontsize=11,ha='center')
        for i,row in enumerate(rows):
            y=35.6-i
            ax.text(0,y+.2,f'{i+1:02d}  {row["name"]}',fontsize=8 if len(row['name'])>30 else 10,va='center')
            rgb=np.array([to_rgba(c)[:3] for c in row['colors']])
            for j,(_,cvd) in enumerate(visions):
                colors=rgb if cvd is None else np.clip(cspace_convert(rgb,{'name':'sRGB1+CVD','cvd_type':cvd,'severity':100},'sRGB1'),0,1)
                for k,color in enumerate(colors):ax.add_patch(Rectangle((4+j*3.5+k*.58,y-.12),.53,.62,facecolor=color,edgecolor='none'))
        fig.savefig(output/'palette_swatches.pdf',bbox_inches='tight',pad_inches=.08)
        plt.close(fig)


def assemble(rows, output):
    import pymupdf as fitz
    combined=fitz.open(); overview=fitz.open(); overview_page=overview.new_page(width=2160,height=2160)
    sheets=fitz.open()
    for i,row in enumerate(rows):
        source=fitz.open(output/row['file'])
        combined.insert_pdf(source)
        if i%6==0:page=sheets.new_page(width=842,height=1191)
        for destination,cols,cellw,cellh,idx in [(overview_page,6,360,360,i),(page,2,421,397,i%6)]:
            x=(idx%cols)*cellw;y=(idx//cols)*cellh
            destination.insert_text((x+9,y+17),f'{i+1:02d}  {row["name"]}',fontsize=11)
            destination.show_pdf_page(fitz.Rect(x+6,y+25,x+cellw-6,y+cellh-6),source,0)
        source.close()
    for doc,name in [(combined,'all_36_corners.pdf'),(overview,'overview_36.pdf'),(sheets,'comparison_sheets.pdf')]:
        doc.save(output/name,garbage=4,deflate=True);doc.close()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--limit',type=int,default=36)
    parser.add_argument('--only',type=int,nargs='+',help='Regenerate selected numbered variants, then assemble existing PDFs')
    parser.add_argument('--input',type=Path,default=Path('validation_runs/nov6a_correlated_v1/analytic_production_production/aggregate_plots/mock_combined_truth_deviation_corner.npz'))
    parser.add_argument('--output',type=Path,default=Path('plots/palette_previews/mock_corners_36'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=published_palettes()[:args.limit]
    with np.load(args.input) as saved:
        values=saved['values'];weights=saved['weights'];names=saved['parameters'];groups=saved['groups']
    labels=[]
    for name in names:
        field=name.split('.',1)[1]
        symbol={'beta_x':r'\beta_x','beta_y':r'\beta_y'}.get(field)
        if symbol is None:symbol=PARAMETER_LABELS.get(field,field).strip('$').replace(r'\log_{10}',r'\log')
        labels.append(r'$\Delta('+symbol+r')/s_{68}$')
    print(f'Building identical contour geometry for {values.shape} saved draws/parameters',flush=True)
    # Retain the figure only in this standalone preview process; production code is unchanged.
    with patch.object(plt,'close'), patch.object(Figure,'savefig'):
        render_block_corner(values,weights,labels,groups,args.output/'unused.pdf',zero_reference=True)
        fig=plt.gcf()
    fig.canvas.draw()
    fixed_bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.04)
    bindings=color_bindings(fig);baseline=geometry_hash(fig)
    for i,row in enumerate(rows):
        row['file']=f'{i+1:02d}_'+row['name'].lower().replace(' ','_')+'.pdf'
        if args.only is not None and i+1 not in args.only:continue
        rgb=np.array([to_rgba(c)[:3] for c in row['colors']])
        for setter,original,matches,scalar,getter in bindings:
            changed=original.copy()
            for k in range(5):changed[matches[:,k],:3]=rgb[k]
            setter(tuple(changed[0]) if scalar else changed)
        assert geometry_hash(fig)==baseline, 'Palette changed contour geometry'
        row['file']=f'{i+1:02d}_'+row['name'].lower().replace(' ','_')+'.pdf'
        with plt.rc_context(fname=STYLE,rc=PAPER_STYLE):fig.savefig(args.output/row['file'],bbox_inches=fixed_bbox)
        for setter,original,matches,scalar,getter in bindings:
            expected=original.copy()
            for k in range(5):expected[matches[:,k],:3]=rgb[k]
            np.testing.assert_allclose(to_rgba_array(getter()),expected,atol=1e-9)
        print(f'{i+1}/36 {row["name"]}',flush=True)
    plt.close(fig)
    swatches(rows,args.output);assemble(rows,args.output)
    manifest=dict(input=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  rows=len(values),parameters=names.tolist(),blocks=list(BLOCK_COLORS),palettes=rows,
                  selection='Five unmodified entries from each published categorical palette, selected in native order with contrast >=3:1 against white and pairwise normal-vision CAM02-UCS distance >=12, chroma 20-42, lightness >=32, max sRGB <=0.96. Published indices recorded. No blending or saturation changes.',
                  geometry_sha256=baseline,geometry_verified_identical_for_all_variants=True,
                  cvd='Machado model via colorspacious; severity 100; clipped to sRGB. Illustration, not universal accessibility certification.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    lines=['# Mock corner palette previews','',
           '36 published palettes; identical saved draws, histogram geometry, contour levels, ranges and transparency. No production defaults changed.',
           '', 'Open `palette_swatches.pdf` first, then `comparison_sheets.pdf` (six corners/page), or `all_36_corners.pdf` (one/page). `overview_36.pdf` is a single vector contact sheet.',
           '', 'Five unmodified published entries, in native order, assigned to Halo, BCG, GRF + GP, scaling relation, sources. All have >=3:1 contrast against white and pairwise CAM02-UCS separation >=12 in normal vision. Pale and gray colors excluded; CAM02-UCS chroma 20-42, lightness >=32, max sRGB <=0.96. No colors numerically modified. Accessibility notes describe the parent palette; the selected subset and transparency are not universally certified.',
           '', '| No. | Palette | Accessibility notes | Source |','|---|---|---|---|']
    lines.extend(f'| {i+1:02d} | [{r["name"]}]({r["file"]}) | {r["accessibility"]} | [Source]({r["source"]}) |' for i,r in enumerate(rows))
    lines.extend(['','Repository snapshot (web checked 2026-11-06): SciencePlots ~9.4k stars, Palettable 816, Colorcet 752, ggsci 751, CMasher 498. These are the projects checked, not a comprehensive popularity ranking. CMasher is mostly continuous maps and was not forced into these categorical previews.'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(args.output/'comparison_sheets.pdf',flush=True)


if __name__=='__main__': main()
