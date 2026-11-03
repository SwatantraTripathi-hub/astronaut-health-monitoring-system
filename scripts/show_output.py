import json
import sys

sys.path.insert(0, ".")

from src.pipeline import analyze

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mitdb/201"
r = analyze(path, data_type="actual_research_data", source_label="check")
print(list(r.keys()))
for ind in r["indicators"]:
    print(ind["name"], "|", ind["status"], "|",
          ind["n_windows_evaluated"], "/", ind["n_windows_total"])
rm = [i for i in r["indicators"] if "RMSSD" in i["name"].upper()]
if rm:
    print(json.dumps(rm[0]["windows"][0], indent=1))
json.dumps(r, allow_nan=False)
print("JSON ok, no NaN")
