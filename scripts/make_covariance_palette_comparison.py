"""Compare eight covariance palettes in one vector PDF without changing the main figure.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.
Rows use the same drawing and checks as make_covariance_figure.py.
"""
import importlib.util
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'covariance', ROOT / 'scripts' / 'make_covariance_figure.py')
drawing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drawing)

# Astrometry, source scatter, stretch/width marker, respectively.
PALETTES = [
    ('A', '#1674A4', '#C44E35', '#B88616'),  # Current: blue, vermilion, gold
    ('B', '#315A85', '#D47763', '#278878'),  # Navy, coral, teal
    ('C', '#287B8E', '#D49A32', '#9B3A55'),  # Ocean blue, amber, burgundy
    ('D', '#387C6D', '#B95173', '#B58A24'),  # Teal, rose, ochre
    ('E', '#5961A8', '#D18A35', '#36836F'),  # Indigo, orange, green
    ('F', '#3278B5', '#BA6748', '#875482'),  # Blue, terracotta, plum
    ('G', '#288C9C', '#A84E72', '#A28326'),  # Cyan, mulberry, olive gold
    ('H', '#6E63A8', '#3B8B7A', '#C16B3E'),  # Violet, teal, copper
]
OUT = ROOT / 'plots' / 'paper' / 'covariance_palette_comparison.pdf'


def make_comparison():
    plt = drawing.plt
    n = len(PALETTES)
    # Preserve each row's physical drawing scale and reserve a slim identification column.
    figure = plt.figure(figsize=(6.65, 2.3*n), facecolor='white')
    add_axes = figure.add_axes
    savefig = figure.savefig
    close = plt.close
    for row, (letter, astrometry, scatter, stretch) in enumerate(PALETTES):
        def row_axes(rect, **kwargs):
            x, y, width, height = rect
            return add_axes([(.35 + 6.3*x)/6.65, (n-row-1+y)/n,
                             6.3*width/6.65, height/n], **kwargs)

        # Reuse the renderer in this row. Suppress its individual save/close calls;
        # the only written artifact is the comparison PDF saved below.
        with patch.multiple(drawing, BLUE=astrometry, ORANGE=scatter, PURPLE=stretch), \
             patch.object(plt, 'figure', return_value=figure), \
             patch.object(plt, 'close'), \
             patch.object(figure, 'add_axes', side_effect=row_axes), \
             patch.object(figure, 'savefig'), \
             patch('builtins.print'):
            drawing.make_figure()
        figure.text(.015, (n-row-1+.56)/n, letter, fontsize=12,
                    fontweight='bold', color=drawing.INK, va='center')
        print(f'{letter}: astrometry {astrometry}; source scatter {scatter}; stretch {stretch}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    savefig(OUT, bbox_inches='tight', pad_inches=.025,
            metadata={'Title': 'Covariance color palette comparison',
                      'Subject': 'Rows A-H: identical illustrative covariance with eight palettes',
                      'Creator': 'scripts/make_covariance_palette_comparison.py'})
    close(figure)
    print(OUT)


if __name__ == '__main__':
    make_comparison()
