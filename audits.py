import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

data_dir = Path(__file__).parent / "data"
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

Image.MAX_IMAGE_PIXELS = None  

def shaof(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect(path):
    try:
        with Image.open(path) as im:      # pass 1: structural check
            im.verify()
        with Image.open(path) as im:      # pass 2: decode all pixels
            im.load()
            return {"width": im.width, "height": im.height,
                    "mode": im.mode, "format": im.format,
                    "readable": True, "error": ""}
    except Exception as e:
        return {"width": None, "height": None, "mode": None, "format": None,
                "readable": False, "error": f"{type(e).__name__}: {e}"[:150]}
        
rows=[]
for path in sorted(data_dir.glob("**/*")):
    if path.is_file():
        rel=path.relative_to(data_dir)
        row = {"path": rel.as_posix(),
                "class": rel.parts[0],
                "extension": path.suffix.lower(),
                "size": round(path.stat().st_size/1024, 2),
                "sha256": shaof(path)}
        row.update(inspect(path))
        rows.append(row)
df=pd.DataFrame(rows)
df.to_csv(output_dir / "audit.csv", index=False)
print(f"Scanned {len(df)} files -> {output_dir}/audit.csv\n")
counts = df.groupby("class").agg(total=("path", "size"),
                                readable=("readable", "sum"))
counts["corrupt"] = counts["total"] - counts["readable"]
counts.loc["TOTAL"] = counts.sum()
print("=== Counts per class ===\n", counts, "\n")
counts.to_csv(output_dir / "counts_per_class.csv")
ok = df[df["readable"]]
stats = ok.groupby("class")[["width", "height", "size"]].agg(["min", "median", "max"])
stats.loc["ALL"] = ok[["width", "height", "size"]].agg(["min", "median", "max"]).T.stack()
print("=== Min / median / max ===\n", stats,"\n")
stats.round(1).to_csv(output_dir / "size_stats.csv")

# print("=== Colour mode ===\n", pd.crosstab(ok["class"], ok["mode"]), "\n")
# print("=== Format ===\n", pd.crosstab(ok["class"], ok["format"]), "\n")
corrupt = df[~df["readable"]]
print(f"=== Corrupt / unreadable files: {len(corrupt)} ===")
if len(corrupt):
    print(corrupt[["path", "extension", "error"]].to_string(index=False))
corrupt.to_csv(output_dir / "corrupt.csv", index=False)

dups = df[df.duplicated("sha256", keep=False)].sort_values("sha256")
n_groups = dups["sha256"].nunique()
n_redundant = len(df) - df["sha256"].nunique()
classes_per_hash = dups.groupby("sha256")["class"].nunique()
n_cross = int((classes_per_hash > 1).sum())

print(f"\n=== Exact duplicates ===")
print(f"Duplicate groups            : {n_groups}")
print(f"Redundant copies            : {n_redundant}")
print(f"Groups in BOTH classes      : {n_cross}")
if len(dups):
    print(dups[["sha256", "class", "path"]].to_string(index=False))
dups.to_csv(output_dir / "duplicates.csv", index=False)

clean = df[df["readable"]].drop_duplicates("sha256")
clean = clean[~clean["sha256"].isin(classes_per_hash[classes_per_hash > 1].index)]
clean.to_csv(output_dir / "clean_index.csv", index=False)
print(f"\nClean images kept for training: {len(clean)}")