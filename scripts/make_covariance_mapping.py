#!/usr/bin/env python
"""Generate the illustrative covariance mapping using Python and TeX.

Run with the project's lenstronomy conda environment.
Only the PDF is saved under plots/paper; compilation files are temporary.
No third-party Python packages or Node.js are required.
The TeX template preserves the reviewed typography, equations and panel layout.
"""
import math
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/paper"
TEMPLATE = Path(__file__).resolve().parent / "templates/covariance_mapping.tex"

# Illustrative values, not measured HST specifications or fitted cluster results.
# At lambda_t=0.05 the image-plane selection, deflection and measurement
# standard deviations are 0.10, 0.05 and 0.03 arcsec, respectively.
LAMBDA_T, LAMBDA_R = 0.05, 0.2
IMAGE_SCALE, SOURCE_SCALE = 15.0, 160.0  # drawing cm per arcsec
ANGLE = math.radians(25)
v, k = (0.10 * LAMBDA_T)**2, (0.05 * LAMBDA_T)**2
sel2 = (v + math.sqrt(v*v + 4*v*k)) / 2
PARAMS = dict(inst=0.01, extra=math.sqrt(0.03**2-0.01**2),
              meas=0.03, res=0.05, sel=math.sqrt(sel2),
              **{'def': 0.0025}, radial=LAMBDA_R)
TRANSITION = PARAMS["sel"] / PARAMS["res"]


def widths(eigenvalue):
    """Source-plane component standard deviations, valid at zero eigenvalue."""
    resolved_fraction = 1 / (1 + (eigenvalue / TRANSITION)**2)
    meas = PARAMS["meas"] * abs(eigenvalue)
    sel = PARAMS["sel"] * math.sqrt(resolved_fraction)
    deflection = PARAMS["def"]
    return dict(meas=meas, sel=sel, **{"def": deflection}, total=math.hypot(meas, sel, deflection))


def validate():
    """Check the variance formula, parity, critical limit and adopted widths."""
    for i in range(1001):
        lam = i / 1000
        w = widths(lam)
        expected = ((PARAMS["meas"]*lam)**2 + PARAMS["def"]**2
                    + PARAMS["sel"]**2 / (1+(lam/TRANSITION)**2))
        if abs(w["total"]**2-expected) > 1e-14:
            raise ValueError("Covariance variance mismatch")
        if widths(-lam) != w:
            raise ValueError("Parity mismatch")
    if not math.isclose(widths(0)["total"],
                        math.hypot(PARAMS["sel"], PARAMS["def"]), abs_tol=1e-14):
        raise ValueError("Critical-limit mismatch")
    for key, target in (("sel", .10), ("def", .05), ("meas", .03)):
        if abs(widths(LAMBDA_T)[key]/LAMBDA_T-target) > 1e-12:
            raise ValueError(f"Adopted {key} image-plane width mismatch")


def point(cx, cy, u, v):
    x = cx + u*math.cos(ANGLE) - v*math.sin(ANGLE)
    y = cy + u*math.sin(ANGLE) + v*math.cos(ANGLE)
    return f"({x:.6f},{y:.6f})"


def ellipse(cx, cy, a, b, color, style, scale):
    thickness = 1.15 if color == "ink" else 0.85 if color == "astro" else 1.1
    return (rf"\draw[{color},line width={thickness}pt,{style},"
            rf"rotate around={{25:({cx},{cy})}}] ({cx},{cy}) ellipse "
            rf"[x radius={a*scale:.16g}cm,y radius={b*scale:.16g}cm];" + "\n")


def contours(cx, cy, image_plane, scale):
    t, r = widths(LAMBDA_T), widths(LAMBDA_R)
    result = ""
    for key, color, style in (("meas", "astro", "densely dotted"),
                              ("sel", "selection", ""),
                              ("def", "floor", "dashed"),
                              ("total", "ink", "")):
        a = t[key] / LAMBDA_T if image_plane else t[key]
        b = r[key] / LAMBDA_R if image_plane else r[key]
        result += ellipse(cx, cy, a, b, color, style, scale)
    return result + rf"\fill[ink] ({cx},{cy}) circle (1pt);" + "\n"


def geometry():
    # Decorative arc, not a ray-traced source. Local tangent is at 25 degrees.
    result = ""
    for width, shade in ((.16, "black!7"), (.105, "black!13"), (.052, "black!21")):
        def side(sign):
            pts = []
            for i in range(121):
                t = -1.7 + 3.4*i/120
                u = 1.25*t
                v = 1.25*(.105*t*t + sign*width*max(0, 1-(t/1.7)**2)**.7)
                pts.append(point(1.5, 2.2, u, v))
            return pts
        result += rf"\fill[{shade}] " + "--".join(side(1)+side(-1)[::-1]) + "--cycle;\n"
    result += contours(1.5, 2.2, True, IMAGE_SCALE)
    result += contours(6, 2.2, False, SOURCE_SCALE)
    radius = SOURCE_SCALE * widths(LAMBDA_T)["total"]
    result += (r"\draw[-{Stealth[length=1.5mm,width=1.1mm]},direction,line width=.8pt] "
               + point(1.5, 2.2, 0, 0) + "--"
               + point(1.5, 2.2, 1.9375, 0) + ";\n")
    result += (r"\node[direction,font=\scriptsize,anchor=south west] at "
               + point(1.5, 2.2, 1.9, .1) + r" {$\boldsymbol e_{\rm t}$};" + "\n")
    result += (r"\draw[-{Stealth[length=1.5mm,width=1.1mm]},direction,line width=.8pt] "
               + point(1.5, 2.2, 0, 0) + "--"
               + point(1.5, 2.2, 0, .8) + ";\n")
    result += (r"\node[direction,font=\scriptsize,anchor=south east] at "
               + point(1.5, 2.2, -.03, .9) + r" {$\boldsymbol e_{\rm r}$};" + "\n")
    result += (r"\draw[-{Stealth[length=1.1mm,width=.85mm]},direction,line width=.85pt] "
               "(6,2.2)--" + point(6, 2.2, radius, 0) + ";\n")
    result += (r"\node[direction,font=\scriptsize,anchor=west] at "
               + point(6, 2.2, radius+.12, .03) + r" {$\sigma_{\rm src,t}$};" + "\n")
    radial_radius = SOURCE_SCALE * widths(LAMBDA_R)["total"]
    result += (r"\draw[-{Stealth[length=1.1mm,width=.85mm]},direction,line width=.85pt] "
               "(6,2.2)--" + point(6, 2.2, 0, radial_radius) + ";\n")
    result += (r"\node[direction,font=\scriptsize,anchor=south east] at "
               + point(6, 2.2, -.03, radial_radius+.10) + r" {$\sigma_{\rm src,r}$};" + "\n")
    return result + (r"\fill[ink] (1.5,2.2) circle (1.2pt);"
                     r"\fill[ink] (6,2.2) circle (1.2pt);" + "\n")


def executable(name):
    found = shutil.which(name)
    if not found and name == "pdflatex":
        candidate = Path("/Library/TeX/texbin/pdflatex")
        if candidate.is_file():
            found = str(candidate)
    if not found:
        raise RuntimeError(f"{name} is required on PATH.")
    return found


def main():
    validate()
    tex = TEMPLATE.read_text()
    replacements = {
        "GEOMETRY": geometry(),
        "XMAX": str(4*TRANSITION),
        "TRANSITION": str(TRANSITION),
        "ASTCOEF": str(PARAMS["meas"]),
        "SELCOEF": str(PARAMS["sel"]),
        "DEFCOEF": str(PARAMS["def"]),
        "YMAX": str(5.7*PARAMS["def"]),
        "MARKER": f'{LAMBDA_T},{widths(LAMBDA_T)["total"]}',
    }
    for name, value in replacements.items():
        tex = tex.replace("@@"+name+"@@", value)
    if "@@" in tex:
        raise ValueError("Unexpanded template token")
    latex = executable("pdflatex")
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "covariance_mapping.pdf"
    with TemporaryDirectory(prefix="covariance_mapping_") as temporary:
        stem = Path(temporary) / "covariance_mapping"
        stem.with_suffix(".tex").write_text(tex)
        compilation = subprocess.run(
            [latex, "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", temporary, str(stem.with_suffix(".tex"))],
            cwd=temporary, capture_output=True, text=True)
        if compilation.returncode:
            raise RuntimeError(compilation.stdout[-5000:] + compilation.stderr)
        shutil.copyfile(stem.with_suffix(".pdf"), output)
    print(output)
    print("Covariance, parity, critical-limit and component-width checks passed.")


if __name__ == "__main__":
    main()
