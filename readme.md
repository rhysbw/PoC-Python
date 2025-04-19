# SportsStats PoC Script
This is a proof-of-concept python script that imports a given CSV file (exported from sportscode) and runs statistical operations on it

## Usage:
- Run the python script with `python main.py --file <path/to/csv/file.csv>`
- This will output a veiaty of statisical analysis

### Example Commands:
#### 1) Default mapping, completed passes only, restrict to "Regular Pass"
python poc_sportstats.py \
  --csv example_csv_data/found_files/Football_Pass_Data_2.csv \
  --team-a "Barcelona" \
  --team-b "Deportivo Alavés" \
  --completed-only \
  --pass-type "Regular Pass" \
  --map mapping_default.yml \
  --ci
#### 2) Defualt mapping, incomplete as well, all pass types
python poc_sportstats.py \
  --csv example_csv_data/found_files/Football_Pass_Data_2.csv \
  --team-a "Barcelona" \
  --team-b "Deportivo Alavés" \
  --map mapping_default.yml \
  --ci

#### 3) Pre‑cleaned dataset with minimal mapping, 99 % CI (alpha 0.01)
python poc_sportstats.py \
  --csv cleaned_data.csv \
  --team-a "Team X" \
  --team-b "Team Y" \
  --map mapping_minimal.yml \
  --alpha 0.01 \
  --ci
