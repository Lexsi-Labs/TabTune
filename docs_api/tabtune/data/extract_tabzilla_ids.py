import re
import sys
import pandas as pd

def extract_ids_from(file_path: str) -> list[int]:
    df = pd.read_csv(file_path, usecols=["dataset_name"])
    def extract_id(name: str):
        base = str(name).split("__fold_")[0]
        m = re.search(r"(\d+)$", base)
        return int(m.group(1)) if m else None
    ids = sorted({i for i in df["dataset_name"].astype(str).map(extract_id) if i is not None})
    return ids

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    ids = extract_ids_from(sys.argv[1])
    print("# TabZilla (OpenML) dataset IDs")
    print("TABZILLA_OPENML_IDS = [")
    for i, id_ in enumerate(ids, 1):
        end = ",\n" if i % 20 == 0 else ", "
        print(f"    {id_}", end=end)
    print("\n]")
