from __future__ import annotations

from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from lenscluster.config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                                MemberPopulationConfig, PriorConfig, RunPathsConfig)


def synthetic_cluster_config(root: Path) -> LensClusterSolverConfig:
    """A four-image, four-member toy cluster: two members near the images, two far away."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images.cat"
    images.write_text("#REFERENCE 3\n1.a 3 5 1 1 0 2 24 1 .05\n1.b -4 -5 1 1 0 2 24 1 .05\n2.a 8 -6 1 1 0 3 24 1 .05\n2.b -7 9 1 1 0 3 24 1 .05\n")
    members = root / "members.cat"
    members.write_text("#REFERENCE 3\ng1 2 3 1 .9 20 20 1 1\ng2 -6 7 1 .8 30 21 1 1\ng3 100 70 1 1 0 21 1 1\ng4 -100 -80 1 1 0 23 1 1\n")
    return LensClusterSolverConfig(
        model=LensModelConfig(
            cosmology=CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
            large_halos=(DPIEHaloConfig(id="halo", x_centre=0.0, y_centre=0.0, angle_pos=0.0, v_disp=600.0, core_radius_kpc=5.0,
                                        cut_radius_kpc=1000.0, z_lens=0.3, ellipticite=0.2,
                                        priors={"v_disp": PriorConfig("uniform", lower=500.0, upper=700.0)}),),
            member_populations=(MemberPopulationConfig(id="members", catalog_path=str(members), mag0=20.0, corekpc=0.2, sigma=100.0,
                                                       cutkpc=10.0, z_lens=0.3,
                                                       sigma_prior=PriorConfig("truncated_normal", mean=100.0, std=20.0, lower=50.0, upper=200.0),
                                                       cutkpc_prior=PriorConfig("truncated_normal", mean=10.0, std=3.0, lower=1.0, upper=50.0)),),
            image_constraints=ImageConstraintsConfig(catalog_path=str(images), sigma_inst_arcsec=0.1)),
        paths=RunPathsConfig(output_dir=str(root / "output"), run_name="toy"))


@pytest.fixture
def synthetic_config(tmp_path: Path) -> LensClusterSolverConfig:
    return synthetic_cluster_config(tmp_path)
