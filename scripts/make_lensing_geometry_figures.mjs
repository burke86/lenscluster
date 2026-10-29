// Synthetic, analytically checked lens-mapping illustrations. No Python needed.
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {pre,post} from './make_covariance_candidates.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'output/pdf/lensing_geometry');fs.mkdirSync(out,{recursive:true});
const T=String.raw,fmt=x=>Number(x.toFixed(6));
const pt=v=>`(${v.map(fmt).join(',')})`;
const text=(x,y,s,opt='')=>`\\node[${opt}] at (${x},${y}){${s}};\n`;
const dot=(x,y)=>`\\fill[ink] (${x},${y}) circle (1.15pt);\n`;
const line=(a,b,opt='')=>`\\draw[${opt}] ${pt(a)}--${pt(b)};\n`;
const arrow=(a,b,c='ink',both=false)=>line(a,b,`${both?'{Stealth[length=1.4mm,width=1mm]}':''}-{Stealth[length=1.4mm,width=1mm]},${c},line width=.85pt`);
const curve=(pts,opt,close=false)=>`\\draw[${opt}] ${pts.map(pt).join('--')}${close?'--cycle':''};\n`;
const ellipse=(x,y,a,b,ang,color='astro',style='')=>`\\draw[${color},line width=1.05pt,${style},rotate around={${ang}:(${x},${y})}] (${x},${y}) ellipse[x radius=${a}cm,y radius=${b}cm];\n`;
const rot=(u,v,a)=>{const t=a*Math.PI/180;return [u*Math.cos(t)-v*Math.sin(t),u*Math.sin(t)+v*Math.cos(t)];};
const offset=(x,y,u,v,a)=>rot(u,v,a).map((z,i)=>z+(i?y:x));
const bar=(x,y,len,label)=>line([x,y],[x+len,y],'ink,line width=.65pt')+text(x+len/2,y+.13,label,'anchor=south,font=\\scriptsize');
const phase=Array.from({length:361},(_,i)=>2*Math.PI*i/360);
let maxBoundaryError=0,maxInverseError=0;
const eps=.12,lt=.16,lr=.8,angle=25,scale=2.3;
for(const t of phase){
 const x=eps/lt*Math.cos(t),y=eps/lr*Math.sin(t);
 maxBoundaryError=Math.max(maxBoundaryError,Math.abs((lt*x)**2+(lr*y)**2-eps**2));
 const bx=eps*lt*Math.cos(t),by=eps*lr*Math.sin(t);
 maxInverseError=Math.max(maxInverseError,Math.abs((bx/lt)**2+(by/lr)**2-eps**2));
}
const figs={};
// Forward and inverse maps are two different experiments, not the same ellipse.
let dual='\\begin{scope}[yshift=1cm]\n';
dual+=ellipse(1.4,3.5,eps*scale,eps*scale,0,'selection','fill=selection!5');
dual+=arrow([2.4,3.5],[4,3.5],'black!55')+text(3.2,3.04,T`$\delta\boldsymbol\theta=\boldsymbol A^{-1}\delta\boldsymbol\beta$`,'font=\\scriptsize');
dual+=ellipse(6.2,3.5,eps/lt*scale,eps/lr*scale,angle,'astro','fill=astro!4');
dual+=arrow([6.2,3.5],offset(6.2,3.5,eps/lt*scale,0,angle),'direction');
dual+=arrow([6.2,3.5],offset(6.2,3.5,0,eps/lr*scale,angle),'floor');
dual+=text(7,4.34,T`$\epsilon/|\lambda_{\rm t}|$`,'direction,font=\\scriptsize');
dual+=text(5.7,3.92,T`$\epsilon/|\lambda_{\rm r}|$`,'floor,font=\\scriptsize');
dual+=text(1.4,2.63,T`$|\delta\boldsymbol\beta|=\epsilon$`)+text(6.2,2.25,T`$\delta\boldsymbol\theta^{\mathsf T}\boldsymbol A^{\mathsf T}\boldsymbol A\,\delta\boldsymbol\theta=\epsilon^2$`);
dual+='\\end{scope}\n';
dual+=ellipse(1.4,.8,eps*scale,eps*scale,0,'astro','fill=astro!4');
dual+=arrow([2.4,.8],[4,.8],'black!55')+text(3.2,.34,T`$\delta\boldsymbol\beta=\boldsymbol A\,\delta\boldsymbol\theta$`,'font=\\scriptsize');
// A larger, explicitly marked scale makes the contracted source visible.
dual+=ellipse(6.2,.8,eps*lt*9,eps*lr*9,angle,'selection','fill=selection!5');
dual+=arrow([6.2,.8],offset(6.2,.8,eps*lt*9,0,angle),'direction');
dual+=arrow([6.2,.8],offset(6.2,.8,0,eps*lr*9,angle),'floor');
dual+=text(6.65,.8,T`$\epsilon|\lambda_{\rm t}|$`,'anchor=west,direction,font=\\scriptsize');
dual+=text(5.55,1.66,T`$\epsilon|\lambda_{\rm r}|$`,'floor,font=\\scriptsize');
dual+=text(1.4,-.13,T`$|\delta\boldsymbol\theta|=\epsilon$`)+text(6.2,-.55,T`$\delta\boldsymbol\beta^{\mathsf T}(\boldsymbol A\boldsymbol A^{\mathsf T})^{-1}\delta\boldsymbol\beta=\epsilon^2$`);
for(const [x,y] of [[1.4,4.5],[6.2,4.5],[1.4,.8],[6.2,.8]])dual+=dot(x,y);
dual+=bar(.3,5.8,.46,T`$0.2''$ source plane`)+bar(5.5,5.8,.46,T`$0.2''$ image plane`);
dual+=bar(.3,1.9,.46,T`$0.2''$ image plane`)+bar(7.5,1.9,.45,T`$0.05''$ source plane`);
figs.lensing_forward_inverse=dual;
// Swap the small eigenvalue; preserve angular scale and source size.
let regimes='';
for(const [i,[t,r]] of [[.8,.8],[.12,.8],[.8,.12]].entries()){
 const x=1.5+4*i,y=2.3,s=1.8,a=eps/t*s,b=eps/r*s;
 regimes+=ellipse(x,y,a,b,25,'astro','fill=astro!4');
 regimes+=ellipse(x,y,eps*s,eps*s,0,'selection','dashed');
 regimes+=arrow([x,y],offset(x,y,a,0,25),'direction');
 regimes+=arrow([x,y],offset(x,y,0,b,25),'floor');regimes+=dot(x,y);
 regimes+=text(x,4.5,`$(\\lambda_{\\rm t},\\lambda_{\\rm r})=(${t},${r})$`);
 regimes+=text(x,.25,`$|\\mu|=${(1/Math.abs(t*r)).toFixed(2)}$`);
 regimes+=text(x,-.25,`Axis ratio $=${(Math.max(t,r)/Math.min(t,r)).toFixed(2)}$`,'font=\\scriptsize');
}
regimes+=bar(.35,5.15,.9,T`$0.5''$`);
regimes+=line([2,-.95],[2.5,-.95],'selection,dashed,thick')+text(2.6,-.95,'Unlensed source','anchor=west,font=\\scriptsize');
regimes+=line([6,-.95],[6.5,-.95],'astro,thick')+text(6.6,-.95,'Local image','anchor=west,font=\\scriptsize');
regimes+=text(5.5,-1.95,T`$a_{\rm t}=\epsilon/|\lambda_{\rm t}|,\quad a_{\rm r}=\epsilon/|\lambda_{\rm r}|,\qquad \mathcal A_{\rm image}=|\mu|\mathcal A_{\rm source}$`);
figs.lensing_stretching_regimes=regimes;
// SIS outer branch: beta = theta - theta_E theta/|theta|.
const b0=[.22,.10],br=Math.hypot(...b0),theta0=b0.map(z=>z*(1+1/br)),phi=Math.atan2(b0[1],b0[0])*180/Math.PI;
const sis=b=>b.map(z=>z*(1+1/Math.hypot(...b)));
let arcs='',maxSIS=0;
for(const [i,size] of [.025,.10].entries()){
 const x=2.3+i*6.2,y=2.8,s=4.7,t=br/(1+br);
 const boundary=phase.map(q=>{const b=[b0[0]+size*Math.cos(q),b0[1]+size*Math.sin(q)],th=sis(b);
  const recovered=th.map(z=>z*(1-1/Math.hypot(...th)));maxSIS=Math.max(maxSIS,Math.hypot(recovered[0]-b[0],recovered[1]-b[1]));
  return [x+s*(th[0]-theta0[0]),y+s*(th[1]-theta0[1])];});
 arcs+=curve(boundary,'astro,line width=1.2pt,fill=astro!5',true);
 arcs+=ellipse(x,y,size*s,size/t*s,phi,'floor','dashed');
 // Tangential eigenvectors evaluated along the mapped source diameter.
 const tangent=rot(0,1,phi);
 for(const u of [-.72,0,.72]){
  const th=sis(b0.map((z,j)=>z+u*size*tangent[j])),a=[x+s*(th[0]-theta0[0]),y+s*(th[1]-theta0[1])];
  const norm=Math.hypot(...th),v=[-th[1]/norm,th[0]/norm];
  arcs+=arrow(a.map((z,j)=>z-Math.min(.25,size*3)*v[j]),a.map((z,j)=>z+Math.min(.25,size*3)*v[j]),'direction',true);
 }
 arcs+=dot(x,y)+text(x,5.8,`$\\epsilon=${size.toFixed(3)}''$`);
 arcs+=bar(x-1.5,.0,.47,T`$0.1''$`);
}
arcs+=line([1.2,-.7],[1.7,-.7],'astro,thick')+text(1.8,-.7,'Exact curved image','anchor=west,font=\\scriptsize');
arcs+=line([5.7,-.7],[6.2,-.7],'floor,dashed,thick')+text(6.3,-.7,'Constant-Jacobian ellipse','anchor=west,font=\\scriptsize');
arcs+=text(5.4,-1.3,T`$\boldsymbol\beta=\boldsymbol\theta-\theta_{\rm E}\boldsymbol\theta/|\boldsymbol\theta|,\quad \theta_{\rm E}=1'',\quad \boldsymbol\beta_0=(0.22'',0.10'')$`);
figs.lensing_arc_curvature=arcs;
// A symmetric, integrable Jacobian whose null eigenvector is not the
// critical-curve tangent. Coordinates dimensionless; map derives from phi.
const b=2,xc=.8,yc=(-1+Math.sqrt(1+4*b**3*xc**2))/(2*b);
const A=[[1+b*yc,b*xc],[b*xc,yc]],eig=[-b*xc/(1+b*yc),1];
const norm=Math.hypot(...eig),ev=eig.map(z=>z/norm),slope=2*b*b*xc/(1+2*b*yc),tv=[1,slope].map(z=>z/Math.hypot(1,slope));
const misalign=Math.acos(Math.abs(ev[0]*tv[0]+ev[1]*tv[1]))*180/Math.PI;
const nullError=Math.hypot(...A.map(row=>row[0]*ev[0]+row[1]*ev[1]));
let critical='';const px=x=>2.6+2*x,py=y=>1.5+2*y;
critical+=curve(Array.from({length:201},(_,i)=>{const x=-1.15+2.9*i/200;return [px(x),py((-1+Math.sqrt(1+4*b**3*x*x))/(2*b))];}),'ink,line width=1.2pt');
const center=[px(xc),py(yc)];
critical+=arrow(center.map((z,j)=>z-1.3*tv[j]),center.map((z,j)=>z+1.3*tv[j]),'floor',true);
critical+=arrow(center.map((z,j)=>z-1.3*ev[j]),center.map((z,j)=>z+1.3*ev[j]),'direction',true);
critical+=dot(...center);
critical+=text(center[0]-.65,center[1]+1.5,T`$\boldsymbol e_{\min}$`,'direction');
critical+=text(center[0]+1.3,center[1]+.7,'Curve tangent','floor,font=\\scriptsize');
critical+=text(.25,1.2,T`$\det\boldsymbol A=0$`,'anchor=west');
critical+=text(3.3,-.15,`$\\angle(\\boldsymbol e_{\\min},\\boldsymbol t_{\\rm crit})=${misalign.toFixed(1)}^\\circ$`);
critical+=text(9.3,3.3,T`$\boldsymbol\beta=\begin{pmatrix}x+bxy\\(b x^2+y^2)/2\end{pmatrix}$`);
critical+=text(9.3,1.95,T`$\boldsymbol A=\begin{pmatrix}1+by&bx\\bx&y\end{pmatrix}$`);
critical+=text(9.3,.6,T`$\det\boldsymbol A=y(1+by)-b^2x^2$`);
critical+=text(6,-1,T`$b=2$; marked point $x=0.8$. Dimensionless local lens map; the inverse map is singular on the curve.`,'font=\\scriptsize');
figs.lensing_critical_direction=critical;
if(maxBoundaryError>1e-12||maxInverseError>1e-12||maxSIS>1e-12||nullError>1e-12)throw Error('Analytic geometry check failed');
for(const [name,body] of Object.entries(figs)){
 fs.writeFileSync(path.join(out,name+'.tex'),pre+body+post);
 try{execFileSync('pdflatex',['-interaction=nonstopmode','-halt-on-error','-output-directory',out,path.join(out,name+'.tex')],{stdio:'pipe'});}catch(e){console.error(e.stdout?.toString().slice(-3000));throw e;}
 execFileSync('pdftoppm',['-scale-to','1700','-singlefile','-png',path.join(out,name+'.pdf'),path.join(out,name)]);
 console.log(path.join(out,name+'.pdf'));
}
fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({maxBoundaryError,maxInverseError,maxSIS,nullError,criticalDirectionAngleDeg:misalign,sourceRadiusArcsec:eps,linearEigenvalues:[lt,lr]},null,2));
fs.writeFileSync(path.join(out,'README.txt'),`Synthetic illustrations, not cluster fits. Regenerate with node scripts/make_lensing_geometry_figures.mjs.\nForward/inverse: epsilon=0.12 arcsec, eigenvalues (0.16,0.8), axes rotated 25 degrees; each panel has its own scale bar.\nStretching regimes: same source size and angular scale in all panels. Purple is e_t, red is e_r.\nCurvature: exact outer SIS branch versus linearization at the image of the source center; purple arrows are local tangential eigenvectors. No critical curve crossing.\nCritical direction: integrable symmetric toy Jacobian; the plotted null eigenvector differs from the curve tangent. At this point no finite inverse ellipse is defined.\nFor a nonsymmetric Jacobian, replace eigenvalue magnitudes with singular values and use the right/left singular vectors in the image/source plane.\n`);
console.log('Analytic boundary, inverse, nonlinear mapping, and null-vector checks passed.');
