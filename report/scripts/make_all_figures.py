"""Regenerate report figures, tables and derived numbers without retraining.

Run from the repository root with its Python environment:
    python report/scripts/make_all_figures.py

Requires generated D2, D3 and D5 banks, frozen models, saved results, and the
original production log. Outputs: figures/, report/tables/, report/provenance/;
inference cache: report/scripts/cache/.
"""

import make_diagnostic_figures
import make_results_figures

if __name__ == "__main__":
    make_results_figures.main()
    make_diagnostic_figures.main()
