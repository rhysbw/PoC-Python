# SPORTSTATS Proof-of-Concept Development Plan

A detailed, modular guide to building a Python-based PoC script for SPORTSTATS, designed to complement Hudl Sportscode.

## Project Overview

### Objective
Build a Python command-line script to:

- Import and process CSV files exported from Hudl Sportscode.
- Conduct statistical analyses (initially a two-proportion z-test).
- Output clear, non-technical interpretations.
- Ensure modularity for future GUI integration.

### Typical Sportscode CSV Columns
- `Timeline`
- `Start time`
- `Duration`
- `Row` (Team)
- `Instance num`
- `Pass Type`
- `Player`
- `Recipient`
- `Half`
- `Outcome`

## Development Timeline

### 1. Repository & Environment Setup (1 hr)
- [ ] Initialize Git repository.
- [ ] Setup Python virtual environment (venv or Poetry).
- [ ] Install dependencies: pandas, numpy, scipy, statsmodels, pytest.
- [ ] Setup linters (flake8, black, isort).

### 2. Data Schema and Mapping (3 hrs)
- [ ] Define default mapping (`schema.py`) for Sportscode fields.
- [ ] Implement user-defined mapping capability via YAML/JSON.
- [ ] Create and test a `ColumnMapper` class for flexible field mappings.

**Example mapping (YAML):**
```yaml
team: Row
event_type: Pass Type
zone: Half
outcome: Outcome
```

### 3. CSV Import & Data Cleaning (4 hrs)
- [ ] Write CSV import function (`io.py`) using pandas.
- [ ] Implement data cleaning functions:
  - Handle missing values.
  - Standardize team names and event types.
  - Validate data consistency.
- [ ] Error handling (malformed files, missing columns).

### 4. Statistical Analysis Module (5 hrs)
- [ ] Implement statistical analysis functions (`stats.py`):
  - Two-proportion z-test to compare events between two teams.
- [ ] Implement basic lag-sequential analysis capability (optional stretch).
- [ ] Ensure output includes statistical significance (p-values).

### 5. Results Interpretation Module (2 hrs)
- [ ] Create natural-language interpreter (`report.py`) for statistical results.
- [ ] Provide readable textual summaries:
  - Example: "Team A's passes in opposition half significantly higher than Team B (p=0.02)."

### 6. Command-Line Interface (CLI) (2 hrs)
- [ ] Develop CLI script (`cli.py`) with argparse.
- Arguments:
  - `--csv`: CSV file input
  - `--map`: Custom mapping file
  - `--team-a` and `--team-b`: Specify teams to compare
  - `--alpha`: Significance level
- [ ] Test CLI usability and output clarity.

### 7. Testing and Quality Assurance (3 hrs)
- [ ] Write pytest unit tests for:
  - Data import and cleaning
  - Statistical tests
  - Output generation
- [ ] Perform manual QA on real datasets.
- [ ] Ensure performance and accuracy.

### 8. Documentation and Reporting (4 hrs)
- [ ] Draft README.md with installation instructions, CLI usage, mapping instructions.
- [ ] Create short technical report (`docs/poc_report.md`):
  - Objectives, methodology, results, recommended next steps.
- [ ] Add CHANGELOG.md for future tracking.

### 9. Final Delivery & Packaging (1 hr)
- [ ] Freeze dependency versions (`requirements.txt`).
- [ ] Tag release version (e.g., v0.1.0) in Git.
- [ ] Deliver project repository/link to client.

## Additional Research & Resources

### Statistical Methods
- **Two-Proportion Z-Test:**
  - Useful for comparing proportions between two independent samples.
  - [Reference - Two-Proportion Z-Test](https://online.stat.psu.edu/stat500/lesson/9/9.2)

- **Lag-Sequential Analysis:**
  - Useful for examining event sequences in performance data.
  - [Reference - Lag-Sequential Analysis in Sports](https://www.tandfonline.com/doi/abs/10.1080/24748668.2020.1743168)

### Python Libraries
- pandas: [https://pandas.pydata.org](https://pandas.pydata.org)
- scipy.stats: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- statsmodels: [https://www.statsmodels.org](https://www.statsmodels.org)

### Sportscode Integration
- Sportscode exports data primarily in CSV/XML formats; direct API or real-time integration is limited.
- Documentation provided in the Sportscode manual sent by Dr Stevens.

## Extension Planning
- **GUI Integration:** Design CLI logic in a modular way that can easily integrate with GUI frameworks like Streamlit or Flask.
- **Additional Statistical Tests:** Allow simple addition of new statistical analysis methods (e.g., Chi-square, permutation tests).

## Definition of Completion
Each section is considered complete when:
- Code passes all unit tests.
- CLI demonstrates required functionality with real or sample data.
- Code is reviewed for style, documentation, and readability.

---

**Estimated Total Development Time:** ~25 hours (50 half-hour increments)

---
# Plan for PoC Python Script
## Pre-requisits
- Use Pandas for dealing with CSV Data.
- Have 1 CSV file (exported from SportsCode)

## High-Level Overview:
1. Import a CSV file
2. Process CSV data:
    1. Extract Important Fields
    2. Layout some schema for interacting with CSV
3. Run Statistic Functions on data:
    1. Simple 3 PoC statistic functions
4. Output in human readable format
5. Infer further statictical Analayis
