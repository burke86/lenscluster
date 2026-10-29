// Temporary illustrations of paper/main.tex's bounded selection covariance.
// Uses Node and TeX; no Python interpreter or environment is needed.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'output/pdf/covariance_candidates');
fs.mkdirSync(out,{recursive:true});
const p={obs:.1,sel:.04,max:.4,floor:.015,radial:.8};
const lc=p.sel/p.max;
const selection=l=>1/Math.sqrt(l*l/p.sel**2+1/p.max**2);
const widths=l=>({imageAst:p.obs,imageSel:selection(l),image:Math.hypot(p.obs,selection(l)),ast:p.obs*Math.abs(l),sel:Math.abs(l)*selection(l),floor:p.floor,total:Math.hypot(p.floor,p.obs*l,l*selection(l))});
// Check the inverse-covariance expression against the draft's scalar formula,
// parity symmetry, and the exact critical limit over all plotted ranges.
for(let i=0;i<=1000;i++){
  const l=i/1000, w=widths(l);
  const expected=p.floor**2+p.obs**2*l*l+p.sel**2*l*l/(l*l+lc*lc);
  if(Math.abs(w.total*w.total-expected)>1e-14)throw Error('Variance mismatch');
  if(widths(-l).total!==w.total)throw Error('Parity mismatch');
}
if(widths(0).total!==p.floor || selection(0)!==p.max)throw Error('Critical limit mismatch');
const pre=String.raw`\documentclass[border=4pt]{standalone}
\usepackage{amsmath,bm,pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta}
\definecolor{astro}{HTML}{1674A4}
\definecolor{selection}{HTML}{218778}
\definecolor{floor}{HTML}{C44E35}
\definecolor{ink}{HTML}{252930}
\definecolor{direction}{HTML}{77529B}
\renewcommand{\familydefault}{\sfdefault}
\pgfplotsset{every axis/.append style={axis lines=left,grid=none,tick align=outside,
 tick label style={font=\scriptsize},label style={font=\small},
 every axis plot/.append style={line width=1pt},legend style={draw=none,font=\scriptsize},
 scaled ticks=false}}
\begin{document}
\begin{tikzpicture}[font=\small,>=Stealth,text=ink]
`;
const post='\n\\end{tikzpicture}\n\\end{document}\n';
function ellipse(cx,cy,a,b,color,style='',scale=1){
 return `\\draw[${color},line width=${color==='ink'?'1.15':'0.85'}pt,${style},rotate around={25:(${cx},${cy})}] (${cx},${cy}) ellipse [x radius=${a*scale}cm,y radius=${b*scale}cm];\n`;
}
function geometry(cx,cy,l,plane,scale){
 const t=widths(l),r=widths(p.radial);
 let s='';
 if(plane==='image'){
  s+=ellipse(cx,cy,t.imageAst,r.imageAst,'astro','densely dotted',scale);
  s+=ellipse(cx,cy,t.imageSel,r.imageSel,'selection','',scale);
  s+=ellipse(cx,cy,t.image,r.image,'ink','',scale);
 }else{
  s+=ellipse(cx,cy,t.ast,r.ast,'astro','densely dotted',scale);
  s+=ellipse(cx,cy,t.sel,r.sel,'selection','',scale);
  s+=ellipse(cx,cy,p.floor,p.floor,'floor','dashed',scale);
  s+=ellipse(cx,cy,t.total,r.total,'ink','',scale);
 }
 return s+`\\fill[ink] (${cx},${cy}) circle (1pt);\n`;
}
const params=String.raw`Illustrative: $\sigma_{\rm obs}=0.10''$, $\sigma_{\rm sel}=0.04''$, $\sigma_{\rm sel,max}=0.40''$, $\sigma_{\rm floor}=0.015''$.`;
const key=String.raw`\draw[astro,densely dotted,thick] (0,0)--(.45,0);\node[anchor=west] at (.5,0){Astrometry};
\draw[selection,thick] (2.7,0)--(3.15,0);\node[anchor=west] at (3.2,0){Selection};
\draw[floor,dashed,thick] (5.2,0)--(5.65,0);\node[anchor=west] at (5.7,0){Source floor};
\draw[ink,thick] (8.2,0)--(8.65,0);\node[anchor=west] at (8.7,0){Total};`;
// A tapered illustrative arc, with its local tangent exactly along the
// low-eigenvalue direction. It is context, not a ray-traced source image.
const tangent=[Math.cos(25*Math.PI/180),Math.sin(25*Math.PI/180)];
const normal=[-tangent[1],tangent[0]];
const point=(cx,cy,u,v)=>`(${(cx+u*tangent[0]+v*normal[0]).toFixed(6)},${(cy+u*tangent[1]+v*normal[1]).toFixed(6)})`;
let schematic='';
for(const [width,shade] of [[.16,'black!7'],[.105,'black!13'],[.052,'black!21']]){
 const side=sign=>Array.from({length:121},(_,i)=>{
  const t=-1.7+3.4*i/120;
  return point(1.5,2.2,t,.105*t*t+sign*width*Math.pow(Math.max(0,1-(t/1.7)**2),.7));
 });
 schematic+=`\\fill[${shade}] ${[...side(1),...side(-1).reverse()].join('--')}--cycle;\n`;
}
schematic+=geometry(1.5,2.2,.05,'image',3.5)+geometry(6,2.2,.05,'source',12);
// Both direction arrows pass through the contour center. The source arrow
// ends at the total contour's tangential semiaxis, not at an arbitrary radius.
const sourceRadius=12*widths(.05).total;
if(Math.abs((sourceRadius/12)**2/widths(.05).total**2-1)>1e-14)throw Error('Width arrow misses contour');
schematic+=`\\draw[{Stealth[length=1.5mm,width=1.1mm]}-{Stealth[length=1.5mm,width=1.1mm]},direction,line width=.8pt] ${point(1.5,2.2,-1.55,0)}--${point(1.5,2.2,1.55,0)};\n`;
schematic+=`\\node[direction,font=\\scriptsize,anchor=south west] at ${point(1.5,2.2,1.52,.08)} {$\\boldsymbol e_{\\rm t}$};\n`;
schematic+=`\\draw[-{Stealth[length=1.1mm,width=.85mm]},direction,line width=.85pt] (6,2.2)--${point(6,2.2,sourceRadius,0)};\n`;
schematic+=`\\node[direction,font=\\scriptsize,anchor=west] at ${point(6,2.2,sourceRadius+.12,.03)} {$w_{\\rm t}$};\n`;
schematic+='\\fill[ink] (1.5,2.2) circle (1.2pt);\\fill[ink] (6,2.2) circle (1.2pt);\n';
schematic+=String.raw`
\draw[-{Stealth[length=2mm,width=1.5mm]},black!55,line width=1.2pt] (3.45,2.2)--(4.65,2.2);
\node[align=center,font=\scriptsize] at (4.05,1.72){$\delta\boldsymbol\beta=\boldsymbol A\delta\boldsymbol\theta$};
\node[font=\scriptsize] at (1.5,.65){$\boldsymbol\Sigma={\color{astro}\sigma_{\rm obs}^{2}\boldsymbol I}+{\color{selection}\boldsymbol\Sigma_{\rm sel}}$};
\node[font=\scriptsize] at (5.5,.65){$\boldsymbol C={\color{astro}\sigma_{\rm obs}^{2}\boldsymbol A\boldsymbol A^{\mathsf T}}+{\color{selection}\boldsymbol A\boldsymbol\Sigma_{\rm sel}\boldsymbol A^{\mathsf T}}+{\color{floor}\sigma_{\rm floor}^{2}\boldsymbol I}$};
\draw[ink] (.2,3.5)--(.9,3.5);\node[above,font=\scriptsize] at (.55,3.5){$0.2''$ image plane};
\draw[ink] (5,3.5)--(5.6,3.5);\node[above,font=\scriptsize] at (5.3,3.5){$0.05''$ source plane};
\begin{axis}[at={(8.3cm,1.1cm)},anchor=south west,width=5cm,height=4.1cm,
 axis line style={-,black!60,line width=.55pt},tick style={black!60},
 xmin=0,xmax=4,ymin=0,ymax=4.2,xtick={0,1,2,3,4},ytick={0,1,2,3,4},
 xlabel={$|\lambda_{\rm t}|/\lambda_{\rm c}$},ylabel={$w_{\rm t}/\sigma_{\rm floor}$}]
\addplot[astro,densely dotted,domain=0:4,samples=101]{2*x/3};
\addplot[selection,domain=0:4,samples=101]{8*x/(3*sqrt(1+x*x))};
\addplot[floor,dashed,domain=0:4]{1};
\addplot[ink,domain=0:4,samples=101]{sqrt(1+(2*x/3)^2+(8*x/3)^2/(1+x*x))};
\addplot[only marks,mark=*,mark options={fill=white,draw=direction,line width=.9pt},mark size=2pt] coordinates {(.5,1.591644852)};
\end{axis}
\begin{scope}[shift={(1.1,-.05)},font=\scriptsize]`+key+String.raw`\end{scope}`;
// Calculate the example marker rather than hard-coding an approximate ordinate.
schematic=schematic.replace('1.591644852',String(widths(.05).total/p.floor));
let response=String.raw`
\begin{axis}[width=6.4cm,height=4.5cm,xmin=0,xmax=5,ymin=0,ymax=1.12,
 xlabel={$|\lambda_a|/\lambda_{\rm c}$},ylabel={$\sigma_{\theta,\rm sel,a}/\sigma_{\rm sel,max}$}]
\addplot[selection,domain=0:5,samples=201]{1/sqrt(1+x*x)};
\addplot[gray,densely dashed,domain=1:5]{1/x};
\node[font=\scriptsize,anchor=west] at (axis cs:.25,1.04){Image-plane cap};
\node[font=\scriptsize,anchor=west,gray] at (axis cs:2.2,.56){$\lambda_{\rm c}/|\lambda_a|$};
\end{axis}
\begin{axis}[at={(7cm,0)},anchor=south west,width=6.4cm,height=4.5cm,xmin=0,xmax=5,ymin=0,ymax=1.8,
 xlabel={$|\lambda_a|/\lambda_{\rm c}$},ylabel={Source-plane uncertainty / $\sigma_{\rm sel}$}]
\addplot[astro,densely dotted,domain=0:5]{x/4};
\addplot[selection,domain=0:5,samples=201]{x/sqrt(1+x*x)};
\addplot[floor,dashed,domain=0:5]{.375};
\addplot[ink,domain=0:5,samples=201]{sqrt(.375^2+(x/4)^2+x*x/(1+x*x))};
\end{axis}
\begin{scope}[shift={(1,-1.35)},font=\scriptsize]`+key+String.raw`\end{scope}
\node[font=\scriptsize] at (6.4,-1.95){`+params+String.raw`};
\node[font=\scriptsize] at (6.4,-2.3){$\lambda_{\rm c}=\sigma_{\rm sel}/\sigma_{\rm sel,max}=0.10$. The critical limit is at the left edge of each panel.};`;
let sequence='';
for(const [i,l] of [.4,.1,.01].entries()){
 const x=2+4*i;
 sequence+=geometry(x,3.3,l,'image',3)+geometry(x,.7,l,'source',10);
 sequence+=`\\node at (${x},4.5){$|\\lambda_{\\rm t}|/\\lambda_{\\rm c}=${Number((l/lc).toPrecision(3))}$};\n`;
}
sequence+=String.raw`
\node[rotate=90] at (-.1,3.3){Image plane};
\node[rotate=90] at (-.1,.7){Source plane};
\draw[->,gray] (3.8,4.5)--(4.4,4.5);\draw[->,gray] (7.8,4.5)--(8.4,4.5);
\draw (10.6,2.3)--(11.2,2.3);\node[below,font=\scriptsize] at (10.9,2.3){$0.2''$};
\draw (10.7,-.4)--(11.2,-.4);\node[below,font=\scriptsize] at (10.95,-.4){$0.05''$};
\begin{scope}[shift={(1,-1)},font=\scriptsize]`+key+String.raw`\end{scope}
\node[font=\scriptsize] at (6,-1.65){`+params+String.raw`};
\node[font=\scriptsize] at (6,-2){Unit-Mahalanobis contours; $\lambda_{\rm r}=0.8$ throughout. Each row uses a fixed angular scale.};`;
export {out,p,lc,widths,pre,post,ellipse,geometry,params,key};
if (process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
for(const [name,body] of Object.entries({covariance_transition:response,covariance_sequence:sequence})){
 if(process.argv.includes('--mapping-only'))throw Error('Use the conda Python interpreter to run scripts/make_covariance_mapping.py');
 fs.writeFileSync(path.join(out,name+'.tex'),pre+body+post);
 execFileSync('/Library/TeX/texbin/pdflatex',['-interaction=nonstopmode','-halt-on-error','-output-directory',out,path.join(out,name+'.tex')],{stdio:'pipe'});
 execFileSync('pdftoppm',['-scale-to','1800','-singlefile','-png',path.join(out,name+'.pdf'),path.join(out,name)],{stdio:'pipe'});
 console.log(path.join(out,name+'.pdf'));
}
fs.writeFileSync(path.join(out,'parameters.json'),JSON.stringify({parameters_arcsec:p,lambda_c:lc,checks:'1001 variance, parity, and critical-limit checks passed; synthetic illustration, not fit results'},null,2)+'\n');
console.log('Verified scalar covariance, parity symmetry, and critical limits.');
}
