"""Optional pytest entry points. These tests have not been run."""
from .validation import (check_selection,check_distances,check_likelihood,
                         check_hessian,check_fields,check_multiplicity,check_priors_and_ratio_anchor,check_compiled_reporting)

def test_catalog_selection(): check_selection()
def test_cosmology(): check_distances()
def test_likelihood_and_source_transform(): check_likelihood()
def test_hessian_direction(): check_hessian()
def test_redshift_fields(): check_fields()
def test_area_and_distinct_images(): check_multiplicity()
def test_physical_uniform_priors_and_anchor(): check_priors_and_ratio_anchor()
def test_compiled_physical_reporting(): check_compiled_reporting()
