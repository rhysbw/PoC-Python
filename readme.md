# SportsStats PoC Script
This is a proof-of-concept python script that imports a given CSV file (exported from sportscode) and runs statistical operations on it

## Usage:
- Run the python script with `python main.py --file <path/to/csv/file.csv>`
- This will output a veiaty of statisical analysis

### Example Commands:
# 1)  All via arguments
python sportstats_logic.py --csv match.csv \
  --group-column Row \
  --groups "Barcelona,Deportivo Alavés" \
  --metric-column Half \
  --metric-success "Opposition Half" \
  --filter "Outcome=Complete" \
  --filter "Pass Type=Regular Pass" \
  --alpha 0.05 --ci

# 2)  Same analysis but via config file
python sportstats_logic.py --csv match.csv --config passes_config.yml
