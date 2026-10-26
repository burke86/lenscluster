"""Normalized scalar priors for posterior diagnostics in sampling coordinates."""
from scipy import stats


def saved_prior(spec):
    """The normalized sampling-coordinate prior, matching distribution_for_site."""
    kind = spec.prior_kind
    if kind == "normal":
        return stats.norm(loc=spec.mean, scale=spec.std)
    if kind == "truncated_normal":
        mean, std = spec.mean, spec.std
        return stats.truncnorm((spec.lower-mean)/std, (spec.upper-mean)/std,
                               loc=mean, scale=std)
    if kind == "uniform":
        return stats.uniform(loc=spec.lower, scale=spec.upper-spec.lower)
    raise ValueError(f"Unsupported saved prior {kind}")

