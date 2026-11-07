CANUCS strong-lensing release for MACS J0416.1-2403 (Rihtarsic et al. 2025, A&A 696, A15), downloaded 2026-10-25 from
https://www.canfar.net/storage/vault/list/canucs/CANUCS_DR1/lensing/macs0416 (CANUCS DR1 lensing; MAST DOI 10.17909/ph4n-6n76).

hlsp_canucs_jwst-hst_multi_macs0416-lenstool-multim_multi_v1_model.txt   gold catalogue, lenstool format: 303 spectroscopic images, 111 families
hlsp_canucs_jwst-hst_multi_macs0416-allmultim-cat_multi_v1_model.txt     full catalogue, 416 images with class G(old)/S(ilver)/B(ronze)/Q
hlsp_canucs_jwst-hst_multi_macs0416-lenstool-clugal_multi_v1_model.txt   cluster members (212; the Bergamini et al. catalogue with the BCG shifted)
hlsp_canucs_jwst-hst_multi_macs0416-lenstool-param_multi_v1_model.txt    lenstool configuration (reference 64.03821668520617 -24.06750122335985)
hlsp_canucs_jwst-hst_multi_macs0416-lenstool-bestparam_multi_v1_model.txt lenstool best-fit model (chi2 344.3 for 354 dof at sigma_pos 0.49")

obs_arcs.cat  = the gold catalogue with a #REFERENCE 0 header and labels recast as FAMILY.letter (D1a.1 -> D1A.a) for the repository reader (id ra dec a b theta z mag; a, b are lenstool shape placeholders, not errors)
members.cat   = the published member catalogue (clugal, 212 members) in the repository's 9-column form (id ra dec a b theta mag d nan),
                token for token, plus galaxy 8971 as the last row, so it sits on the member scaling relation. The CANUCS model fits
                8971 as its own halo (potential 6); its position here is that halo's centre, (13.33912", 2.62495") from the CANUCS
                reference. Its F160W magnitude, 19.6191, is from the Bergamini et al. photometry that every clugal magnitude copies.

members.yaml  = members.cat as a YAML member catalog (src/lenscluster/member_catalog.py; scripts/convert_member_catalog.py), the
                one run.xsh M0416_CANUCS2025 reads: band F160W, the two BCGs (2526 NE, 3158 SW) with role bcg, and the
                MUSE stellar velocity dispersions of 52 members (sigma_ap in a 0.8" aperture) from granata2026_m0416_kinematics.txt,
                matched by position within 0.6" (median 0.24"). Members with kinematics are protected from the impact cut.
                It reads to the same model as members.cat (identical likelihood and gradient at the start).

granata2026_m0416_kinematics.txt = the M0416 table of Granata et al. 2026, A&A (arXiv:2603.26869v3), Appendix C: 52
                members with id, RA, Dec, MUSE id, sigma, err (km/s) and S/N, transcribed from the arXiv PDF on
                2026-11-02 (the same table is at CDS, which was unreachable then). BCG-N is id 1 (275.6 +- 1.5 km/s),
                BCG-S id 73 (286.4 +- 2.1 km/s).

images.yaml   = the gold catalogue in the lenscluster YAML image format (scripts/convert_canucs_catalog.py): 83 systems, 15 with
                knots, 111 knot-families, 303 images; system/knot/image from the published labels (K6c.2 = system K6, knot C,
                image b); positions and z from the lenstool input (catalogue + about 20 mas, the frame of members.cat);
                K6e images 1 and 3 renumbered by position to match the other K6 knots. run.xsh M0416_CANUCS2025 reads it.
