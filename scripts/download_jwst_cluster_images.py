#!/home/dutra/.conda/envs/lenstronomy/bin/python
"""Download public NIRCam mosaics and M0416 redshift catalogs.

Run with the lenstronomy Python interpreter; requires curl and astropy.
Defaults cover RGB=(F444W,F200W,F115W) and RGB=(F200W,F150W,F115W).
Default destination: /data/sync/GalaxyClusterImaging/JWST_Images, alongside BUFFALO_Images.
Downloads original science mosaics plus background- and bright-galaxy-subtracted
CANUCS images for MACS0416. No corresponding subtracted product is listed in the
selected AS1063 DAWN release; that combination is explicitly skipped.
Reproject to a common WCS before combining filters, especially SW with LW.
M0416 redshift catalogs are included by default; --catalogs-only skips images.
Catalogs go to ../RedshiftCatalogs/M0416 beside the image directories.
Use --refresh-catalogs to re-fetch checked releases, preserving changed copies.
See m0416_redshift_sources.json and M0416_REDSHIFT_CATALOGS.md for coverage.

Sources:
https://dawn-cph.github.io/dja/imaging/v7/ (AS1063, Grizli v7.5)
https://archive.stsci.edu/hlsp/canucs/ (MACS0416, CANUCS DR1)
https://archive.stsci.edu/hlsps/canucs/dr1/hlsp_canucs_readme.txt
"""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


CATALOG_MANIFEST = Path(__file__).with_name("m0416_redshift_sources.json")


def catalog_sources(selected=None):
    manifest = json.loads(CATALOG_MANIFEST.read_text())
    sources = manifest["sources"]
    if selected:
        unknown = set(selected) - {source["id"] for source in sources}
        if unknown:
            raise ValueError(f"Unknown catalog sources: {sorted(unknown)}")
        sources = [source for source in sources if source["id"] in selected]
    return manifest, sources


def validate_catalog(path, product, directory):
    """Validate tables separately from image FITS; retain native columns and flags."""
    from astropy.io import fits
    from astropy.table import Table

    opener = gzip.open if product["filename"].endswith(".gz") else open
    fmt = product["format"]
    with opener(path, "rb") as stream:
        data = stream.read(4096) if fmt == "csv" else stream.read()
    if not data.strip():
        raise ValueError("Empty catalog response")
    if data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError("Archive returned an HTML page instead of a catalog")
    if fmt == "fits":
        with fits.open(path, memmap=False) as hdus:
            hdus.verify("exception")
            tables = [dict(hdu=h.name, rows=len(h.data), columns=list(h.columns.names))
                      for h in hdus if isinstance(h, (fits.BinTableHDU, fits.TableHDU))]
        if not tables or not any(t["rows"] for t in tables):
            raise ValueError("No nonempty FITS table")
    elif fmt == "csv":
        # Stream the full DJA compilation rather than holding its many columns
        # and decompressed text simultaneously in memory. Reading to EOF checks CRC.
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            columns = next(reader)
            rows = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(columns):
                    raise ValueError(f"CSV row {rows + 2} has inconsistent column count")
                rows += 1
        if not rows or len(columns) < 2:
            raise ValueError("Empty or malformed CSV catalog")
        tables = [dict(rows=rows, columns=columns)]
    elif fmt == "cds":
        # CDS selects the ReadMe schema using the actual table filename,
        # not our transfer's .part suffix.
        with tempfile.TemporaryDirectory() as temp:
            table_path = Path(temp) / product["filename"].removesuffix(".gz")
            table_path.write_bytes(data)
            table = Table.read(table_path, format="ascii.cds",
                               readme=str(directory / "ReadMe"))
        if len(table) == 0 or len(table.colnames) < 2:
            raise ValueError("Empty or malformed catalog table")
        tables = [dict(rows=len(table), columns=table.colnames)]
    else:
        data.decode("utf-8")  # Original ASCII catalogs and documentation.
        tables = []
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest, tables=tables)


def download_catalog(product, directory, *, refresh=False):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / product["filename"]
    if destination.exists() and not refresh:
        metadata = validate_catalog(destination, product, directory)
    else:
        partial = destination.with_name(destination.name + ".part")
        # Small catalog transfers start fresh. Resuming a mutable release can
        # accidentally join bytes from two different versions.
        subprocess.run([
            "curl", "--fail", "--location", "--show-error", "--progress-bar",
            "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
            "--max-time", "1800", "--output", str(partial), product["url"],
        ], check=True)
        metadata = validate_catalog(partial, product, directory)
        if destination.exists():
            with destination.open("rb") as stream:
                old_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if old_digest != metadata["sha256"]:
                archive = directory / ".versions" / old_digest
                archive.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, archive / destination.name)
                sidecar = destination.with_name(destination.name + ".json")
                if sidecar.exists():
                    shutil.copy2(sidecar, archive / sidecar.name)
        partial.replace(destination)
    record = dict(product, **metadata, path=str(destination.resolve()),
                  verified_utc=datetime.now(timezone.utc).isoformat())
    sidecar = destination.with_name(destination.name + ".json")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(sidecar)
    return record


def download_catalogs(output, sources, manifest, *, refresh=False):
    """Try every requested source; report failures without claiming completeness."""
    root = output / "M0416"
    root.mkdir(parents=True, exist_ok=True)
    report = dict(checked_date=manifest["checked_date"], scope=manifest["scope"],
                  coverage_notes=manifest["coverage_notes"], files=[], failures=[])
    for source in sources:
        for entry in source["files"]:
            product = dict(entry, source_id=source["id"], release=source["release"],
                           source_page=source["source_page"],
                           redshift_kind=source["redshift_kind"], notes=source["notes"])
            print(f"Catalog: {source['id']} / {entry['filename']}", flush=True)
            try:
                report["files"].append(download_catalog(product, root / source["id"], refresh=refresh))
            except (OSError, EOFError, ValueError, csv.Error, subprocess.CalledProcessError) as error:
                report["failures"].append(dict(product, error=str(error)))
                print(f"FAILED: {source['id']} / {entry['filename']}: {error}", flush=True)
    report["complete"] = not report["failures"]
    temporary = root / "download_manifest.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(root / "download_manifest.json")
    print(f"Catalogs: {len(report['files'])} ready, {len(report['failures'])} failed. {root}")
    return report["complete"]


FILTERS = ("F115W", "F150W", "F200W", "F444W")
DEFAULT_OUTPUT = Path("/data/sync/GalaxyClusterImaging/JWST_Images")


def products(clusters, filters, m0416_sw_mas, product_types=("sci", "bgsub-sci")):
    for cluster in clusters:
        for band in filters:
            if cluster == "AS1063":
                filename = f"abells1063-grizli-v7.5-{band.lower()}-clear_drc_sci.fits.gz"
                base = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/"
                release = "DAWN/Grizli v7.5"
            else:
                scale = 40 if band == "F444W" else m0416_sw_mas
                filename = f"hlsp_canucs_jwst_nircam_macs0416-clu-{scale}mas_{band.lower()}_v1_sci.fits.gz"
                base = "https://archive.stsci.edu/hlsps/canucs/dr1/macs0416/clu/"
                release = "CANUCS DR1; DOI 10.17909/18nv-np70"
            for product_type in product_types:
                if cluster == "AS1063" and product_type != "sci":
                    continue
                name = filename.replace("_sci.fits.gz", f"_{product_type}.fits.gz")
                yield dict(cluster=cluster, filter=band, filename=name,
                           url=base + name, release=release, product_type=product_type)


def validate(path):
    """Check gzip integrity, FITS structure, and celestial WCS; record provenance."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales

    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass  # Read through the trailer to verify gzip length and CRC.
    with fits.open(path, memmap=False) as hdus:
        hdus.verify("exception")
        hdu = next(h for h in hdus if h.header.get("NAXIS") == 2)
        header = hdu.header
        wcs = WCS(header)
        if not wcs.has_celestial:
            raise ValueError(f"Missing celestial WCS: {path}")
        shape = [int(header["NAXIS2"]), int(header["NAXIS1"])]
        scales = np.asarray(proj_plane_pixel_scales(wcs.celestial)) * 3600
        if not np.all(np.isfinite(scales) & (scales > 0)):
            raise ValueError(f"Invalid pixel scale: {path}")
        metadata = dict(shape=shape, pixel_scale_arcsec=scales.tolist(),
                        bunit=header.get("BUNIT"), fits_filter=header.get("FILTER"))
    with path.open("rb") as stream:
        metadata["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata["bytes"] = path.stat().st_size
    return metadata


def download(product, output):
    directory = output / product["cluster"]
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / product["filename"]
    if destination.exists():
        print(f"Validate existing: {destination}", flush=True)
        metadata = validate(destination)
    else:
        partial = destination.with_name(destination.name + ".part")
        print(f"Download: {product['cluster']} {product['filter']} {product['product_type']}", flush=True)
        # Partial transfers survive interruptions and resume on the next invocation.
        subprocess.run([
            "curl", "--fail", "--location", "--show-error", "--progress-bar",
            "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
            "--speed-limit", "1024", "--speed-time", "120",
            "--continue-at", "-", "--output", str(partial), product["url"],
        ], check=True)
        print(f"Validate FITS and gzip: {partial.name}", flush=True)
        metadata = validate(partial)
        partial.replace(destination)
    record = dict(product, **metadata, path=str(destination.resolve()),
                  verified_utc=datetime.now(timezone.utc).isoformat())
    sidecar = destination.with_name(destination.name + ".json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(sidecar)
    print(f"Ready: {destination} ({metadata['bytes'] / 1e6:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", nargs="+", choices=("AS1063", "M0416"),
                        default=["AS1063", "M0416"])
    parser.add_argument("--filters", nargs="+", choices=FILTERS, default=list(FILTERS))
    parser.add_argument("--products", nargs="+", choices=("sci", "bgsub-sci"),
                        default=["sci", "bgsub-sci"],
                        help="Default: original and subtracted science images where available")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination directory (default: /data/sync/GalaxyClusterImaging/JWST_Images)")
    parser.add_argument("--m0416-sw-mas", type=int, choices=(20, 40), default=20,
                        help="Short-wavelength pixel scale; F444W is always 40 mas")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print URLs and destinations without network access or file creation")
    parser.add_argument("--catalogs-only", action="store_true", help="Download M0416 catalogs only; skip images")
    parser.add_argument("--no-redshift-catalogs", action="store_true", help="Download images only")
    parser.add_argument("--catalog-output", type=Path,
                        help="Default: OUTPUT's parent / RedshiftCatalogs")
    parser.add_argument("--catalog-sources", nargs="+", help="Subset of source IDs in m0416_redshift_sources.json")
    parser.add_argument("--refresh-catalogs", action="store_true",
                        help="Fetch catalog releases again; archive changed existing files")
    args = parser.parse_args()
    want_catalogs = "M0416" in args.clusters and not args.no_redshift_catalogs
    if args.catalogs_only and not want_catalogs:
        parser.error("--catalogs-only requires M0416 and cannot use --no-redshift-catalogs")
    if (args.catalog_sources or args.refresh_catalogs) and not want_catalogs:
        parser.error("Catalog options require M0416 redshift catalogs enabled")
    try:
        manifest, sources = catalog_sources(args.catalog_sources)
    except ValueError as error:
        parser.error(str(error))
    catalog_output = args.catalog_output or args.output.parent / "RedshiftCatalogs"
    if not args.catalogs_only and "AS1063" in args.clusters and "bgsub-sci" in args.products:
        print("Unavailable: AS1063 bgsub-sci is not listed in DAWN/Grizli v7.5; skipping.")
    plan = [] if args.catalogs_only else list(products(dict.fromkeys(args.clusters), dict.fromkeys(args.filters),
                         args.m0416_sw_mas, dict.fromkeys(args.products)))
    if not plan and not want_catalogs:
        parser.error("No available products for this selection")
    if args.dry_run:
        for product in plan:
            print(f"{product['url']}\n  -> {args.output / product['cluster'] / product['filename']}")
        if want_catalogs:
            print(f"M0416 redshift releases checked {manifest['checked_date']}; {len(sources)} sources")
            for source in sources:
                print(f"[{source['id']}] {source['release']} ({source['redshift_kind']})")
                for product in source["files"]:
                    print(f"{product['url']}\n  -> {catalog_output / 'M0416' / source['id'] / product['filename']}")
            print(manifest["scope"])
        return
    if shutil.which("curl") is None:
        parser.error("curl is required")
    # Fail before transferring files if FITS validation dependencies are missing.
    import astropy.io.fits  # noqa: F401

    for product in plan:
        download(product, args.output)
    if want_catalogs:
        if not download_catalogs(catalog_output, sources, manifest, refresh=args.refresh_catalogs):
            raise SystemExit("Some catalogs failed; see M0416/download_manifest.json. Rerun to retry.")
    if plan:
        print("Complete. RGB: R=F444W G=F200W B=F115W; sharp RGB: R=F200W G=F150W B=F115W.")
        print("Align filters to a common WCS before creating RGB images.")


if __name__ == "__main__":
    main()
