# Rossmann Demand Forecasting

- **Versions:** use current stable versions and verify them (`pip index versions <package>`, the release page for
  GitHub Actions); do not write versions from memory. The default `python` on this machine is 3.9: use Python 3.14
  (`py -3.14`) and a fresh virtual environment. After a bump, compare the test results and the numbers from
  `python analysis/backtest_recursive.py` with `analysis/backtest_results.json`; a version change should not move them.
- **Requirements:** `requirements-render.txt` is what the hosted dashboard and CI install; `requirements.txt` is the
  full training stack. Keep both pinned.
- **Tests:** `python -m pytest` needs no Kaggle data.
- **Data:** raw Kaggle CSVs, `models/*.pkl` and logs stay out of git. `deploy_data/` is the committed bundle the app
  runs on; regenerate it with `build_deploy_data.py` after retraining.
- **Commits:** keep code changes and documentation changes in separate commits.
- **Docs:** every number in the README or `docs/METHODOLOGY.md` should trace to a committed file (`metrics.json`,
  `cv_results.csv`, `analysis/*.csv|json`). `docs/WALKTHROUGH.md` is local-only (excluded in `.git/info/exclude`).
