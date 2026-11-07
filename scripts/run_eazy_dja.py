"""Photometric redshifts with eazy-py for a DJA (DAWN/grizli) photometric catalog that has no photo-z release, e.g.
abells1063-grizli-v7.5-fix_phot_apcorr.fits.  The fit follows the CANUCS / DJA practice: tweak_fsps_QSF_12_v3 templates,
total fluxes from the 0.5" aperture (<band>_tot_1, uJy), 5% systematic error, no magnitude prior.

eazy needs its filter and template files: eazy.fetch_eazy_photoz() clones them into the package; this script links
them into --workdir.

Example:
    python scripts/run_eazy_dja.py --catalog .../abells1063-grizli-v7.5-fix_phot_apcorr.fits --workdir .../eazy_run
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

# EAZY FILTER.RES.latest numbers for the DJA band names (u suffix = WFC3/UVIS).
FILTER_NUMBERS = {"f090w": 363, "f105w": 202, "f110w": 241, "f115w": 364, "f125w": 203, "f140w": 204, "f150w": 365, "f160w": 205,
                  "f200w": 366, "f250m": 379, "f277w": 375, "f300m": 380, "f350lpu": 339, "f356w": 376, "f390wu": 210, "f410m": 383,
                  "f435w": 233, "f444w": 377, "f475w": 234, "f480m": 386, "f606w": 236, "f606wu": 214, "f625w": 237, "f775w": 238,
                  "f814w": 239, "f850lp": 240}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--aperture", default="1", help="DJA aperture index of the total fluxes (_tot_<i>)")
    p.add_argument("--z-max", type=float, default=12.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--templates", default="templates/fsps_full/tweak_fsps_QSF_12_v3.param",
                   help="template set relative to --workdir; templates/sfhz/blue_sfhz_13.param has strong emission lines")
    p.add_argument("--specz", default="", help="CSV with ra, dec, z (secure spec-z) matched within 0.3\" for the zero-point iteration")
    p.add_argument("--iterate-zp", type=int, default=0, help="iterations of the per-band zero-point corrections fitted on the spec-z sources")
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    import eazy
    import eazy.photoz
    data = Path(eazy.__file__).parent / "data" / "eazy-photoz"
    if not data.exists():
        eazy.fetch_eazy_photoz()
    for name in ("filters", "templates"):
        if not (work / name).exists():
            os.symlink(data / name, work / name)
    os.chdir(work)

    t = Table(fits.open(args.catalog)[1].data)
    keep = ["id", "ra", "dec"]
    translate = ["z_spec z_spec"]
    for band, number in FILTER_NUMBERS.items():
        f, e = f"{band}_tot_{args.aperture}", f"{band}_etot_{args.aperture}"
        if f in t.colnames and e in t.colnames and np.isfinite(t[f]).any():
            keep += [f, e]
            translate += [f"{f} F{number}", f"{e} E{number}"]
    sub = t[keep]
    sub["z_spec"] = -1.0
    if args.specz:
        import pandas as pd
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        s = pd.read_csv(args.specz)
        idx, d, _ = SkyCoord(s["ra"].to_numpy() * u.deg, s["dec"].to_numpy() * u.deg).match_to_catalog_sky(
            SkyCoord(np.asarray(sub["ra"]) * u.deg, np.asarray(sub["dec"]) * u.deg))
        ok = d.arcsec < 0.3
        sub["z_spec"][idx[ok]] = s["z"].to_numpy()[ok]
        print(f"[eazy] {ok.sum()} spec-z matched within 0.3 arcsec")
    for c in keep[3:]:
        sub[c] = np.where(np.isfinite(sub[c]), sub[c], -99.0)              # EAZY's missing value
    sub.write("eazy_input.fits", overwrite=True)
    Path("eazy.translate").write_text("\n".join(translate) + "\n")
    print(f"[eazy] {len(sub)} sources, {len(translate) // 2} bands")

    params = {"CATALOG_FILE": "eazy_input.fits", "MAIN_OUTPUT_FILE": "eazy_output",
              "FILTERS_RES": "filters/FILTER.RES.latest", "TEMPLATES_FILE": args.templates,
              "APPLY_PRIOR": "n", "PRIOR_ABZP": 23.9, "Z_MIN": 0.01, "Z_MAX": args.z_max, "Z_STEP": 0.01, "Z_STEP_TYPE": 1,
              "SYS_ERR": 0.05, "N_MIN_COLORS": 3, "CAT_HAS_EXTCORR": "y", "MW_EBV": 0.0}
    pz = eazy.photoz.PhotoZ(param_file=None, translate_file="eazy.translate", params=params, n_proc=args.threads)
    pz.fit_catalog(n_proc=args.threads)
    if args.iterate_zp:
        # Per-band multiplicative corrections from the residuals of the spec-z sources fitted at their spec-z (EAZY's own
        # iteration); the corrections are applied and the catalog refitted.
        pz.iterate_zp_templates(idx=None, update_templates=False, update_zeropoints=True, iter=args.iterate_zp, n_proc=args.threads,
                                save_templates=False, error_residuals=False, prior=False, get_spatial_offset=False)
        print("[eazy] zero-point corrections: " + ", ".join(f"{name} {zp:.3f}" for name, zp in zip(pz.flux_columns, pz.zp)))
        pz.fit_catalog(n_proc=args.threads)
    zout, _ = pz.standard_output(save_fits=True, prior=False, beta_prior=False)
    print(f"[eazy] done: {Path('eazy_output.zout.fits').resolve()}; {np.isfinite(zout['z_phot']).sum()} finite z_phot")


if __name__ == "__main__":
    main()
