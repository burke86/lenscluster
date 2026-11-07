"""Render the aggregate GRF/GP recovery from saved posterior field diagnostics.

For updated field calculations, use plot_mock_posterior_fields.py.
"""
from pathlib import Path
import argparse
from lenscluster.mock_validation.aggregate_plots import load_campaign
from lenscluster.mock_validation.field_recovery_plots import plot_field_recovery

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    try:
        campaign = load_campaign(args.campaign)
    except ValueError:
        campaign = load_campaign(args.campaign, converged_only=False)
    result = plot_field_recovery(campaign,args.campaign/'aggregate_plots')
    if result is None:
        parser.error('No posterior field arrays found; run plot_mock_posterior_fields.py first')
    print(result)
