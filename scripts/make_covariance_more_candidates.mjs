// Five additional synthetic covariance illustrations. Run with node.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {out,p,lc,widths,pre,post,ellipse,geometry,params,key} from './make_covariance_candidates.mjs';
const raw=String.raw;
const num=x=>Number(x.toPrecision(4));
const node=(x,y,s,options='')=>`\\node[${options}] at (${x},${y}){${s}};\n`;
const footer=(x,y,extra)=>node(x,y,params,'font=\\scriptsize')+node(x,y-.38,extra,'font=\\scriptsize');
const legend=(x,y)=>`\\begin{scope}[shift={(${x},${y})},font=\\scriptsize]${key}\\end{scope}\n`;
const axis=(x,options)=>`\\begin{axis}[at={(${x}cm,0)},anchor=south west,width=6.4cm,height=4.5cm,${options}]\n`;
const end='\\end{axis}\n';
function curve(fn,color,style='',max=5){
 const coords=Array.from({length:301},(_,i)=>{const x=max*i/300;return `(${x.toFixed(6)},${fn(x).toFixed(9)})`;}).join(' ');
 return `\\addplot[${color},${style}] coordinates {${coords}};\n`;
}
const figs={};

// Equal image displacements mapped with the same symmetric Jacobian as the contours.
const l=.05,d=.12,angle=25*Math.PI/180;
const e=[Math.cos(angle),Math.sin(angle)],n=[-e[1],e[0]];
let residual=geometry(2,2.4,l,'image',4)+geometry(7.6,2.4,l,'source',15);
function arrow(x,y,v,amount,scale,color,label,anchor){
 const a=x+v[0]*amount*scale,b=y+v[1]*amount*scale;
 return `\\draw[->,${color},line width=1.2pt] (${x},${y})--(${a},${b});\n`+node(a,b,label,`font=\\scriptsize,${color},anchor=${anchor}`);
}
residual+=arrow(2,2.4,e,d,4,'direction',raw`$\delta\theta_{\rm t}$`,'north west');
residual+=arrow(2,2.4,n,d,4,'gray',raw`$\delta\theta_{\rm r}$`,'south east');
residual+=arrow(7.6,2.4,e,d*l,15,'direction',raw`$\delta\beta_{\rm t}$`,'north west');
residual+=arrow(7.6,2.4,n,d*p.radial,15,'gray',raw`$\delta\beta_{\rm r}$`,'south east');
residual+=raw`\draw[->,gray,thick] (4.1,2.4)--(5.5,2.4);
\node[font=\scriptsize] at (4.8,1.9){$\delta\boldsymbol\beta=\boldsymbol A_i\delta\boldsymbol\theta$};
\draw (.6,4.1)--(1.4,4.1);\node[above,font=\scriptsize] at (1,4.1){$0.2''$ image plane};
\draw (6.9,4.1)--(7.65,4.1);\node[above,font=\scriptsize] at (7.275,4.1){$0.05''$ source plane};`;
const qt=(d*l/widths(l).total)**2,qr=(d*p.radial/widths(p.radial).total)**2;
residual+=node(2,.35,raw`$|\delta\theta_{\rm t}|=|\delta\theta_{\rm r}|=0.12''$`);
residual+=node(7.6,.35,raw`$|\delta\beta_{\rm t}|=0.006'',\quad |\delta\beta_{\rm r}|=0.096''$`);
residual+=node(11.6,3.1,raw`$Q=\delta\boldsymbol\beta^{\mathsf T}\boldsymbol C_i^{-1}\delta\boldsymbol\beta$`);
residual+=node(11.6,2.3,`$Q_{\\rm t}=${qt.toFixed(3)}$`,'direction');
residual+=node(11.6,1.7,`$Q_{\\rm r}=${qr.toFixed(3)}$`,'gray');
residual+=legend(1.5,-.35)+footer(6.7,-1,raw`$\lambda_{\rm t}=0.05$, $\lambda_{\rm r}=0.8$. $Q$ is the quadratic residual penalty at fixed covariance, not the full likelihood.`);
figs.covariance_equal_displacements=residual;

// Five fixed-scale snapshots, including the exact singular-Jacobian limit.
let critical='';
for(const [i,t] of [4,1,.3,.1,0].entries()){
 const x=1.5+3.4*i;
 critical+=geometry(x,3.3,t*lc,'image',2.7)+geometry(x,.65,t*lc,'source',9);
 critical+=node(x,4.5,`$|\\lambda_{\\rm t}|/\\lambda_{\\rm c}=${t}$`);
 if(i<4)critical+=`\\draw[->,gray] (${x+1.25},4.5)--(${x+2.05},4.5);\n`;
}
critical+=raw`\node[rotate=90] at (-.15,3.3){Image plane};
\node[rotate=90] at (-.15,.65){Source plane};
\draw (15.6,2.2)--(16.14,2.2);\node[below,font=\scriptsize] at (15.87,2.2){$0.2''$};
\draw (15.6,-.45)--(16.05,-.45);\node[below,font=\scriptsize] at (15.825,-.45){$0.05''$};`;
critical+=legend(3.5,-1.15)+footer(8.2,-1.8,raw`$\lambda_{\rm r}=0.8$; fixed scale within each row. At $\lambda_{\rm t}=0$, mapped components collapse to lines; the total retains width $\sigma_{\rm floor}$.`);
figs.covariance_critical_sequence=critical;

// The precise increment from selection, on identical source-plane scales.
let comparison='';
for(const [i,t] of [4,1,.1].entries()){
 const x=2+4*i,l=t*lc,w=widths(l),r=widths(p.radial),scale=12;
 comparison+=ellipse(x,1.8,Math.hypot(p.floor,p.obs*l),Math.hypot(p.floor,p.obs*p.radial),'gray','dashed',scale);
 comparison+=ellipse(x,1.8,w.total,r.total,'ink','',scale);
 comparison+=node(x,3.3,`$|\\lambda_{\\rm t}|/\\lambda_{\\rm c}=${t}$`);
 comparison+=node(x,.25,`$w_{\\rm t}/w_{\\rm t,0}=${(w.total/Math.hypot(p.floor,p.obs*l)).toFixed(2)}$`);
}
comparison+=raw`\draw[gray,dashed,thick] (2.2,-.4)--(2.65,-.4);\node[anchor=west,font=\scriptsize] at (2.7,-.4){Astrometry + source floor};
\draw[ink,thick] (7,-.4)--(7.45,-.4);\node[anchor=west,font=\scriptsize] at (7.5,-.4){With selection};
\draw (11.1,2.2)--(11.7,2.2);\node[above,font=\scriptsize] at (11.4,2.2){$0.05''$};`;
comparison+=footer(6,-1.05,raw`Source-plane unit-Mahalanobis contours; $\lambda_{\rm r}=0.8$. $w_{\rm t,0}^{2}=\sigma_{\rm floor}^{2}+\sigma_{\rm obs}^{2}\lambda_{\rm t}^{2}$.`);
figs.covariance_selection_comparison=comparison;

// Use a common unnormalized eigenvalue axis: normalizing each by its own
// lambda_c would hide the shift in transition that this figure illustrates.
const caps=[.2,.4,.8],styles=['densely dotted','solid','dashed'];
let cap=axis(0,raw`xmin=0,xmax=.5,ymin=0,ymax=.86,xlabel={$|\lambda_a|$},ylabel={$\sigma_{\theta,\rm sel,a}$ [arcsec]}`);
for(const [i,m] of caps.entries()){
 cap+=curve(l=>1/Math.sqrt(l*l/p.sel**2+1/m**2),'selection',styles[i],.5);
 cap+=`\\addplot[only marks,mark=o,selection,mark size=2pt] coordinates {(${p.sel/m},${m/Math.sqrt(2)})};\n`;
}
cap+=end+axis(7,raw`xmin=0,xmax=.5,ymin=0,ymax=.043,xlabel={$|\lambda_a|$},ylabel={$\sigma_{\beta,\rm sel,a}$ [arcsec]}`);
for(const [i,m] of caps.entries()){
 cap+=curve(l=>p.sel*l/Math.sqrt(l*l+(p.sel/m)**2),'selection',styles[i],.5);
 cap+=`\\addplot[only marks,mark=o,selection,mark size=2pt] coordinates {(${p.sel/m},${p.sel/Math.sqrt(2)})};\n`;
}
cap+=end;
for(const [i,m] of caps.entries()){
 const x=.4+i*4.5;
 cap+=`\\draw[selection,${styles[i]},thick] (${x},-1.35)--(${x+.45},-1.35);\n`+node(x+.55,-1.35,`$\\sigma_{\\rm sel,max}=${m.toFixed(2)}''$`,'anchor=west,font=\\scriptsize');
}
cap+=node(6.5,-1.95,raw`Illustrative selection term only; $\sigma_{\rm sel}=0.04''$ is fixed. Circles mark $|\lambda_a|=\lambda_{\rm c}$.`,'font=\\scriptsize');
cap+=node(6.5,-2.33,raw`Increasing the image-plane cap moves $\lambda_{\rm c}=\sigma_{\rm sel}/\sigma_{\rm sel,max}$ toward zero.`,'font=\\scriptsize');
figs.covariance_cap_variation=cap;

const variances=t=>{const l=t*lc;return [(p.obs*l)**2,(l/Math.sqrt(l*l/p.sel**2+1/p.max**2))**2,p.floor**2];};
const colors=['astro','selection','floor'],lineStyles=['densely dotted','solid','dashed'];
let variance=axis(0,raw`xmin=0,xmax=8,ymin=0,ymax=5.5,xlabel={$|\lambda_a|/\lambda_{\rm c}$},ylabel={Variance / $\sigma_{\rm sel}^{2}$}`);
for(let j=0;j<3;j++)variance+=curve(t=>variances(t)[j]/p.sel**2,colors[j],lineStyles[j],8);
variance+=curve(t=>variances(t).reduce((a,b)=>a+b)/p.sel**2,'ink','',8);
variance+=end+axis(7,raw`xmin=0,xmax=8,ymin=0,ymax=1.05,ytick={0,.25,.5,.75,1},xlabel={$|\lambda_a|/\lambda_{\rm c}$},ylabel={Fraction of total variance}`);
for(let j=0;j<3;j++)variance+=curve(t=>{const v=variances(t);return v[j]/v.reduce((a,b)=>a+b);},colors[j],lineStyles[j],8);
variance+=end+legend(1,-1.35)+footer(6.4,-1.95,raw`The three variance terms add to $\sigma_{\rm src,a}^{2}$; their fractional contributions sum to one.`);
figs.covariance_variance_budget=variance;

// Verify a rotated matrix construction and residual quadratic forms independently.
const A=[[l*e[0]**2+p.radial*n[0]**2,l*e[0]*e[1]+p.radial*n[0]*n[1]],
 [l*e[0]*e[1]+p.radial*n[0]*n[1],l*e[1]**2+p.radial*n[1]**2]];
const C=[[widths(l).total**2*e[0]**2+widths(p.radial).total**2*n[0]**2,
 widths(l).total**2*e[0]*e[1]+widths(p.radial).total**2*n[0]*n[1]],
 [0,widths(l).total**2*e[1]**2+widths(p.radial).total**2*n[1]**2]];
C[1][0]=C[0][1];
const det=C[0][0]*C[1][1]-C[0][1]**2;
for(const [v,expected] of [[e,qt],[n,qr]]){
 const b=A.map(row=>d*(row[0]*v[0]+row[1]*v[1]));
 const q=(C[1][1]*b[0]**2-2*C[0][1]*b[0]*b[1]+C[0][0]*b[1]**2)/det;
 if(Math.abs(q-expected)>1e-12)throw Error('Matrix residual check failed');
}
for(let i=0;i<=1000;i++){
 const t=8*i/1000,v=variances(t),sum=v.reduce((a,b)=>a+b);
 if(Math.abs(sum-widths(t*lc).total**2)>1e-14)throw Error('Budget does not match total');
 if(Math.abs(v.reduce((a,b)=>a+b/sum,0)-1)>1e-14)throw Error('Fractions do not sum to one');
}
for(const m of caps){
 const c=p.sel/m;
 if(Math.abs(1/Math.sqrt(c*c/p.sel**2+1/m**2)-m/Math.sqrt(2))>1e-14)throw Error('Cap transition mismatch');
}
for(const [name,body] of Object.entries(figs)){
 fs.writeFileSync(path.join(out,name+'.tex'),pre+body+post);
 try{execFileSync('/Library/TeX/texbin/pdflatex',['-interaction=nonstopmode','-halt-on-error','-output-directory',out,path.join(out,name+'.tex')],{stdio:'pipe'});}
 catch(error){console.error(error.stdout?.toString().slice(-2500));throw error;}
 execFileSync('pdftoppm',['-scale-to','1800','-singlefile','-png',path.join(out,name+'.pdf'),path.join(out,name)]);
 console.log(path.join(out,name+'.pdf'));
}
fs.writeFileSync(path.join(out,'additional_checks.json'),JSON.stringify({parameters:p,quadratic_penalties:{tangential:qt,radial:qr},caps,checks:'Rotated matrix residuals, 1001 variance sums and fraction sums, cap transitions, and inherited critical-limit checks passed.'},null,2)+'\n');
console.log('All additional covariance checks passed.');
