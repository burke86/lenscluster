#!/home/dutra/.conda/envs/lenstronomy/bin/python
"""The imaging and truth maps the figures need, downloaded and kept in one tree, data/imaging/<cluster>/<source>/.

    download_imaging.py download [--clusters hera ares m0416 as1063] [--kinds jwst ffsims hst] [--hst-scale 60|30]
                                 [--local-source DIR] [--ffsims-source DIR] [--copy] [--dry-run]
    download_imaging.py zip [--out FILE] [--kinds jwst ffsims] [--with-catalogs]
    download_imaging.py unpack ARCHIVE
    download_imaging.py list

Sources.  JWST: CANUCS DR1 NIRCam mosaics of MACS J0416 (MAST HLSP, 20 mas SW and 40 mas F444W) and the DAWN/Grizli
v7.5 mosaics of Abell S1063; the four RGB bands of run.xsh (F115W, F150W, F200W, F444W).  HST: BUFFALO v1.0 drizzled
mosaics of both clusters in the seven HFF bands (60 mas by default; 30 mas on request, 2.7 GB per band).  FF-SIMS: the
simulated HST images (F435W, F606W, F814W) and the true kappa and shear maps at z = 9 of Hera and Ares (Meneghetti et
al. 2017).  The FF-SIMS site (pico.oabo.inaf.it/~massimo/Public/FF) has been offline since 2026 and the Wayback Machine
holds only its text files, so those FITS come from --ffsims-source (an unpacked zip of this script, or the old
data/ff_sims/published tree); the original URLs are kept in the manifest and tried first.

Files already present under --local-source (a synced copy with the same file names, e.g.
/data/sync/GalaxyClusterImaging) are linked instead of downloaded; --copy copies them.  Every file is validated (gzip
CRC, FITS structure, WCS for the mosaics) and gets a .json sidecar with its origin and sha256; data/imaging/manifest.json
lists the tree.  `zip` packs the JWST files, the FF-SIMS files and the FF-SIMS site mirror (no HST; with the catalogs/ tree by
default) into one archive that `unpack` restores.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "imaging"
JWST_BANDS = ("F115W", "F150W", "F200W", "F444W")
HFF_BANDS = ("F435W", "F606W", "F814W", "F105W", "F125W", "F140W", "F160W")
FF_BANDS = ("F435W", "F606W", "F814W")
FF_TRUTH = ("kappa_z9_0.fits", "gammax_z9_0.fits", "gammay_z9_0.fits")
CANUCS = "https://archive.stsci.edu/hlsps/canucs/dr1/macs0416/clu/"
DJA = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/"
BUFFALO = "https://archive.stsci.edu/hlsps/buffalo/{cluster}/images/{scale}mas/"
FFSIMS = "http://pico.oabo.inaf.it/~massimo/Public/FF/"
WAYBACK = "http://web.archive.org/web/2025id_/"
# The whole FF-SIMS site, for posterity: every file its pages link to, per simulated cluster, plus the pages themselves.
# Hera and Ares have the lens results (deflection angles, F/G maps, kappa and shear at z = 9); Hermes and Hephaestus
# only the simulated images and catalogs.  finalshearHST.cat of Hera and Ares is linked but was never archived.
FF_SITE_FILES = {
    "hera": ["fits/hera/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshearHST.cat", "finalshear_z.cat", "galcat2.txt",
                                        "multimages.reg", "multimages.txt", "simulation_subaru_R.fits")]
            + [f"fits/hera/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS]
            + ["results/hera/" + n for n in ("deflAnglesHera.fits", "Hera_F.fits", "Hera_G.fits") + FF_TRUTH],
    "ares": ["fits/ares/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshear.cat", "finalshearHST.cat",
                                        "multimages.reg", "multimages.txt", "simulation_subaru_R.fits")]
            + [f"fits/ares/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS]
            + ["results/ares/" + n for n in ("deflAnglesAres.fits", "Ares_F.fits", "Ares_G.fits") + FF_TRUTH],
    "hermes": ["fits/hermes/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshear.cat", "nome.reg", "nome.txt", "simulation_subaru_R.fits")]
              + [f"fits/hermes/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS],
    "hephaestus": ["fits/hephaestus/" + n for n in ("clgal_cat.reg", "clgal_cat.txt", "finalshear.cat", "nome.reg", "nome.txt", "simulation_subaru_R.fits")]
                  + [f"fits/hephaestus/simulation_hst_{b.lower()}.fits" for b in HFF_BANDS],
}
FF_SITE_PAGES = ["index.html", "instructions.html", "contacts.html", "hst_image.html", "subaru_image.html", "multiple_images.html",
                 "cluster_galaxies.html", "ares.html", "hera.html", "hermes.html", "hephaestus.html", "style.css",
                 "images/ares_thumb.jpg", "images/hera_thumb.jpg", "images/hermes_thumb.jpg", "images/hephaestus_thumb.jpg",
                 "images/udf_banner2.jpeg", "images/img03.png", "images/img04.jpg"]
# Our complete copy of the site (wget mirror) lives at /data/pico.oabo.inaf.it/~massimo/Public/FF on the Linux machine;
# the only files it lacks are finalshearHST.cat (Hera, Ares) and nome.txt / nome.reg (Hermes, Hephaestus), which the site
# linked but, as far as the Wayback Machine knows, never served.
FF_MIRROR = Path("/data/pico.oabo.inaf.it/~massimo/Public/FF")


def manifest(hst_scale: int = 60) -> list[dict]:
    """Every product: where it goes (relpath under data/imaging), where it comes from, what it is for."""
    out = []
    for band in JWST_BANDS:
        scale = 40 if band == "F444W" else 20
        name = f"hlsp_canucs_jwst_nircam_macs0416-clu-{scale}mas_{band.lower()}_v1_sci.fits.gz"
        out.append(dict(cluster="m0416", source="canucs_dr1", kind="jwst", band=band, relpath=f"m0416/canucs_dr1/{name}",
                        urls=[CANUCS + name], release="CANUCS DR1 (Rihtarsic et al. 2025); DOI 10.17909/18nv-np70", use="rgb"))
        name = f"abells1063-grizli-v7.5-{band.lower()}-clear_drc_sci.fits.gz"
        out.append(dict(cluster="as1063", source="dja_v7.5", kind="jwst", band=band, relpath=f"as1063/dja_v7.5/{name}",
                        urls=[DJA + name], release="DAWN JWST Archive, Grizli v7.5 mosaics", use="rgb"))
    for cluster, token in (("m0416", "macs0416"), ("as1063", "abells1063")):
        for band in HFF_BANDS:
            camera = "acs-wfc" if band in FF_BANDS else "wfc3-ir"
            name = f"hlsp_buffalo_hst_{camera}-{hst_scale}mas_{token}_{band.lower()}_v1.0_drz.fits"
            out.append(dict(cluster=cluster, source="buffalo", kind="hst", band=band, relpath=f"{cluster}/buffalo/{hst_scale}mas/{name}",
                            urls=[BUFFALO.format(cluster=token, scale=hst_scale) + name],
                            release="BUFFALO v1.0 (Steinhardt et al. 2020); DOI 10.17909/t9-k61t-qc38", use="rgb"))
    for cluster in ("hera", "ares"):
        for band in FF_BANDS:
            name = f"simulation_hst_{band.lower()}.fits"
            out.append(dict(cluster=cluster, source="ffsims", kind="ffsims", band=band, relpath=f"{cluster}/ffsims/{name}",
                            urls=[FFSIMS + f"fits/{cluster}/{name}", WAYBACK + FFSIMS + f"fits/{cluster}/{name}"],
                            release="FF-SIMS (Meneghetti et al. 2017, MNRAS 472, 3177)", use="rgb"))
        for name in FF_TRUTH:
            out.append(dict(cluster=cluster, source="ffsims", kind="ffsims", band=None, relpath=f"{cluster}/ffsims/{name}",
                            urls=[FFSIMS + f"results/{cluster}/{name}", WAYBACK + FFSIMS + f"results/{cluster}/{name}"],
                            release="FF-SIMS (Meneghetti et al. 2017, MNRAS 472, 3177)", use="truth"))
    # The site mirror: ffsims_site/<path as on the site>.  The Hera and Ares files the figures use appear twice in the
    # manifest (here and above); `zip` stores them once and `unpack` links the <cluster>/ffsims copies to the mirror.
    for cluster, paths in FF_SITE_FILES.items():
        for path in paths:
            out.append(dict(cluster=cluster, source="ffsims_site", kind="ffsims_site", band=None, relpath=f"ffsims_site/{path}",
                            urls=[FFSIMS + path, WAYBACK + FFSIMS + path],
                            release="FF-SIMS site mirror (Meneghetti et al. 2017)", use="archive"))
    for path in FF_SITE_PAGES:
        out.append(dict(cluster="ffsims", source="ffsims_site", kind="ffsims_site", band=None, relpath=f"ffsims_site/{path}",
                        urls=[FFSIMS + path, WAYBACK + FFSIMS + path], release="FF-SIMS site mirror (Meneghetti et al. 2017)", use="archive"))
    return out


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path: Path, product: dict) -> dict:
    """gzip CRC, FITS structure, a 2-d image with a celestial WCS for the mosaics (the FF-SIMS maps have a plain
    linear WCS); returns the metadata of the sidecar."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    if product["kind"] == "ffsims_site":                     # the mirror: FITS must open, anything else is kept as is
        if path.suffix.lower() == ".fits":
            with fits.open(path, memmap=False) as hdus:
                hdus.verify("exception")
                shape = [int(n) for n in hdus[0].shape] if hdus[0].data is not None else []
        else:
            shape = []
        return dict(shape=shape, bytes=path.stat().st_size, sha256=sha256(path))
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as stream:
            while stream.read(8 << 20):
                pass
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        hdu = next(h for h in hdus if h.header.get("NAXIS") == 2)
        header = hdu.header
        wcs = WCS(header)
        if product["kind"] != "ffsims" and not wcs.has_celestial:
            raise ValueError(f"{path}: no celestial WCS")
        scales = np.asarray(proj_plane_pixel_scales(wcs)) * (3600.0 if product["kind"] != "ffsims" or wcs.has_celestial else 1.0)
        meta = dict(shape=[int(header["NAXIS2"]), int(header["NAXIS1"])], pixel_scale_arcsec=[float(s) for s in scales],
                    bunit=header.get("BUNIT"), fits_filter=header.get("FILTER"))
    meta.update(bytes=path.stat().st_size, sha256=sha256(path))
    return meta


def curl(url: str, partial: Path) -> bool:
    result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3", "--retry-delay", "3",
                             "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-",
                             "--output", str(partial), url])
    return result.returncode == 0


def find_local(product: dict, roots: list[Path]) -> Path | None:
    """The same file name under any of the roots (synced copies keep the archive names)."""
    name = Path(product["relpath"]).name
    cluster = product.get("cluster")
    for root in roots:
        if not root.exists():
            continue
        found = [c for c in root.rglob(name) if c.is_file()]
        # Site files share names across clusters (clgal_cat.txt): the copy must sit under a folder named for the cluster.
        if product["kind"] in ("ffsims", "ffsims_site"):
            found = [c for c in found if cluster in {part.lower() for part in c.parts}]
        if found:
            return found[0]
    return None


def fetch(product: dict, *, local: list[Path], copy: bool, dry_run: bool) -> dict | None:
    destination = ROOT / product["relpath"]
    sidecar = destination.with_name(destination.name + ".json")
    if destination.exists():
        if sidecar.exists():
            print(f"present   {product['relpath']}")
            return json.loads(sidecar.read_text())
        print(f"validate  {product['relpath']}")
        meta, origin = validate(destination, product), {"origin": "present"}
    else:
        source = find_local(product, local)
        if dry_run:
            print(f"{'link' if source else 'download':9s} {product['relpath']}  <-  {source or product['urls'][0]}")
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            if copy:
                print(f"copy      {product['relpath']}  <-  {source}")
                shutil.copy2(source, destination)
            else:
                print(f"link      {product['relpath']}  ->  {source}")
                destination.symlink_to(source.resolve())
            origin = {"origin": str(source.resolve())}
        else:
            partial = destination.with_name(destination.name + ".part")
            for url in product["urls"]:
                print(f"download  {product['relpath']}  <-  {url}", flush=True)
                if "web.archive.org" in url:
                    time.sleep(2.0)                               # the Wayback Machine rate-limits bursts (HTTP 429)
                if curl(url, partial):
                    origin = {"origin": url}
                    break
                partial.unlink(missing_ok=True)
            else:
                print(f"MISSING   {product['relpath']}: no source answered (give --local-source or --ffsims-source)")
                return None
            partial.replace(destination)
        meta = validate(destination, product)
    record = dict(product, **meta, **origin, path=str(destination), verified_utc=datetime.now(timezone.utc).isoformat())
    sidecar.write_text(json.dumps(record, indent=1) + "\n")
    return record


def write_manifest(records: list[dict]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifest.json").write_text(json.dumps(dict(written_utc=datetime.now(timezone.utc).isoformat(), root=str(ROOT),
                                                       files=records), indent=1) + "\n")
    (ROOT / "README.md").write_text(README)


def selected(args) -> list[dict]:
    """--clusters narrows the figure data; the site mirror (Hermes, Hephaestus, the pages) always comes whole."""
    return [p for p in manifest(args.hst_scale) if p["kind"] in args.kinds
            and (p["cluster"] in args.clusters or p["kind"] == "ffsims_site")]


def cmd_download(args) -> int:
    local = [Path(p) for p in (args.local_source or [])] + [Path(p) for p in (args.ffsims_source or [])]
    if FF_MIRROR.exists() and FF_MIRROR not in local:
        local.append(FF_MIRROR)
    records, missing = [], 0
    for product in selected(args):
        record = fetch(product, local=local, copy=args.copy, dry_run=args.dry_run)
        if record is None and not args.dry_run:
            missing += 1
        elif record is not None:
            records.append(record)
    if not args.dry_run:
        present = [json.loads(p.read_text()) for p in sorted(ROOT.rglob("*.json")) if p.name != "manifest.json"]
        write_manifest(present)
        print(f"{len(records)} ready, {missing} missing; manifest {ROOT / 'manifest.json'}")
    return 1 if missing else 0


def cmd_zip(args) -> int:
    out = Path(args.out)
    products = [p for p in manifest() if p["kind"] in args.kinds and (p["cluster"] in args.clusters or p["kind"] == "ffsims_site")]
    members: list[tuple[Path, str]] = []
    links: dict[str, str] = {}
    missing = 0
    for product in products:
        path = ROOT / product["relpath"]
        if product["kind"] == "ffsims" and "ffsims_site" in args.kinds:
            # Stored once, in the mirror; unpack recreates <cluster>/ffsims/<name> as a link into it.
            folder = "results" if Path(product["relpath"]).name in FF_TRUTH else "fits"
            links[product["relpath"]] = f"ffsims_site/{folder}/{product['cluster']}/{Path(product['relpath']).name}"
            continue
        if not path.exists():
            print(f"missing   {product['relpath']} (not retrievable; left out)" if product["kind"] == "ffsims_site"
                  else f"MISSING   {product['relpath']} (run download first)")
            missing += 1
            if product["kind"] != "ffsims_site":
                return 1
            continue
        members.append((path, f"imaging/{product['relpath']}"))
        sidecar = path.with_name(path.name + ".json")
        if sidecar.exists():
            members.append((sidecar, f"imaging/{product['relpath']}.json"))
    if args.mirror_dir:
        # The whole local copy of the FF-SIMS site, verbatim, in place of the manifest's ffsims_site entries.
        mirror = Path(args.mirror_dir)
        members = [m for m in members if not m[1].startswith("imaging/ffsims_site/")]
        for path in sorted(mirror.rglob("*")):
            if path.is_file() and ".comments" not in path.parts:
                members.append((path, f"imaging/ffsims_site/{path.relative_to(mirror)}"))
    for name in ("README.md", "manifest.json"):
        if (ROOT / name).exists():
            members.append((ROOT / name, f"imaging/{name}"))
    links_file = ROOT / "links.json"
    links_file.write_text(json.dumps(links, indent=1) + "\n")
    members.append((links_file, "imaging/links.json"))
    if args.with_catalogs:
        for path in sorted((REPO / "catalogs").rglob("*")):
            if path.is_file() and not path.name.endswith((".fits", ".fits.gz")):
                members.append((path, f"catalogs/{path.relative_to(REPO / 'catalogs')}"))
    total = sum(p.stat().st_size for p, _ in members)
    print(f"{len(members)} files, {total / 1e9:.2f} GB -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", allowZip64=True) as archive:
        for path, arcname in members:
            # gzipped mosaics do not shrink again; the plain FITS of FF-SIMS and the text files do.
            method = zipfile.ZIP_STORED if path.name.endswith((".gz", ".jpg", ".jpeg")) else zipfile.ZIP_DEFLATED
            archive.write(path, arcname, compress_type=method)
            print(f"  {arcname}", flush=True)
    print(f"written {out} ({out.stat().st_size / 1e9:.2f} GB); {len(links)} figure files are links into the mirror; "
          f"{missing} site files were not retrievable")
    return 0


def cmd_unpack(args) -> int:
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        for name in names:
            target = (REPO / "data" / name) if name.startswith("imaging/") else (REPO / name)
            if target.exists() and not args.force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 16 << 20)
            print(f"  {target.relative_to(REPO)}")
    links_file = ROOT / "links.json"
    if links_file.exists():
        for relpath, mirror in json.loads(links_file.read_text()).items():
            target = ROOT / relpath
            if not target.exists() and (ROOT / mirror).exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.path.relpath(ROOT / mirror, target.parent))
                print(f"  {target.relative_to(REPO)} -> {mirror}")
    print(f"unpacked {len(names)} entries under {ROOT} (and catalogs/)")
    return 0


def cmd_list(args) -> int:
    for product in manifest(args.hst_scale):
        path = ROOT / product["relpath"]
        state = "link" if path.is_symlink() else "file" if path.exists() else "-"
        print(f"{state:5s} {product['kind']:6s} {product['relpath']}")
    return 0


README = """# data/imaging

The mosaics and truth maps the figures read, one folder per cluster and source (scripts/download_imaging.py):

    hera/ffsims/      simulation_hst_f435w|f606w|f814w.fits, kappa_z9_0 / gammax_z9_0 / gammay_z9_0.fits (FF-SIMS, Meneghetti et al. 2017)
    ares/ffsims/      the same
    m0416/canucs_dr1/ hlsp_canucs_jwst_nircam_macs0416-clu-{20,40}mas_<band>_v1_sci.fits.gz, F115W F150W F200W F444W (CANUCS DR1)
    m0416/buffalo/    <scale>mas/hlsp_buffalo_hst_*_macs0416_<band>_v1.0_drz.fits, the seven HFF bands (HST, BUFFALO v1.0)
    as1063/dja_v7.5/  abells1063-grizli-v7.5-<band>-clear_drc_sci.fits.gz, F115W F150W F200W F444W (DAWN JWST Archive)
    as1063/buffalo/   as for m0416
    ffsims_site/      the FF-SIMS site as it was (pages, thumbnails, fits/<cluster>/ and results/<cluster>/ for Hera, Ares,
                      Hermes and Hephaestus); the site went offline in 2026 and the Wayback Machine kept only its text files

Each file has a .json sidecar (origin URL or local path, sha256, shape, pixel scale) and manifest.json lists them.
run.xsh points ImageCatalogCutoutConfig.image_dir at <cluster>/<source> and the truth maps at <cluster>/ffsims.
Files may be symlinks into a synced copy (download --local-source); `zip` follows them.  In an archive the Hera and
Ares figure files are stored once under ffsims_site/ and links.json says where; `unpack` recreates the links.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    clusters = dict(nargs="+", choices=("hera", "ares", "m0416", "as1063"), default=["hera", "ares", "m0416", "as1063"])
    d = sub.add_parser("download", help="fetch (or link) the files into data/imaging")
    d.add_argument("--clusters", **clusters)
    d.add_argument("--kinds", nargs="+", choices=("jwst", "ffsims", "ffsims_site", "hst"), default=["jwst", "ffsims", "ffsims_site"],
                   help="jwst, ffsims and the FF-SIMS site mirror by default; hst adds the BUFFALO mosaics (0.7 GB per band at 60 mas)")
    d.add_argument("--hst-scale", type=int, choices=(60, 30), default=60)
    d.add_argument("--local-source", nargs="+", help="roots searched for already downloaded copies of the same file names")
    d.add_argument("--ffsims-source", nargs="+", help="roots holding the FF-SIMS files (an unpacked zip, data/ff_sims/published)")
    d.add_argument("--copy", action="store_true", help="copy local sources instead of symlinking them")
    d.add_argument("--dry-run", action="store_true")
    d.set_defaults(run=cmd_download)
    z = sub.add_parser("zip", help="pack the JWST and FF-SIMS files (no HST) and the catalogs into one archive")
    z.add_argument("--out", default=str(REPO / "data" / "lenscluster_imaging_jwst_ffsims.zip"))
    z.add_argument("--clusters", **clusters)
    z.add_argument("--kinds", nargs="+", choices=("jwst", "ffsims", "ffsims_site", "hst"), default=["jwst", "ffsims", "ffsims_site"])
    z.add_argument("--with-catalogs", action=argparse.BooleanOptionalAction, default=True)
    z.add_argument("--mirror-dir", default=str(FF_MIRROR) if FF_MIRROR.exists() else None,
                   help="pack this local copy of the FF-SIMS site verbatim as imaging/ffsims_site (default: the wget mirror when present)")
    z.set_defaults(run=cmd_zip)
    u = sub.add_parser("unpack", help="restore an archive made by zip into data/imaging and catalogs/")
    u.add_argument("archive")
    u.add_argument("--force", action="store_true", help="overwrite files that exist")
    u.set_defaults(run=cmd_unpack)
    ls = sub.add_parser("list", help="the manifest and what is present")
    ls.add_argument("--hst-scale", type=int, choices=(60, 30), default=60)
    ls.set_defaults(run=cmd_list)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
