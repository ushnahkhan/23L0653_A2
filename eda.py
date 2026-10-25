from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from termcolor import COLORS
datadir=Path("data")
figdir=Path("output/figures")
figdir.mkdir(parents=True, exist_ok=True)
Image.MAX_IMAGE_PIXELS = None  
df=pd.read_csv(Path("output/clean_index.csv"))
classes=sorted(df["class"].unique())
colors={classes[0]:"tab:orange", classes[1]:"tab:blue"}
seed=42
fig,axes=plt.subplots(len(classes), 8, figsize=(2.2*8, 2.6*len(classes)))

for r,cls in enumerate(classes):
    sample=df[df["class"]==cls].sample(8, random_state=seed)
    for c,(_,row) in enumerate(sample.iterrows()):
        ax=axes[r,c]
        with Image.open(datadir/row["path"]) as im:
            im=im.convert("RGB")
            im.thumbnail((300,300))
            ax.imshow(im)
        ax.set_title(f"{int(row['width'])}x{int(row['height'])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[r,0].set_ylabel(cls, fontsize=13,fontweight="bold")

fig.suptitle("Sample images from each class", fontsize=16,fontweight="bold")
fig.tight_layout()
fig.savefig(figdir/"sample_grid.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
features = [("width", "Width (pixels)"),
            ("height", "Height (pixels)"),
            ("size", "File size (KB, log scale)")]

for ax, (col, label) in zip(axes, features):
    values = df[col]
    if col == "size":
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
        ax.set_xscale("log")
    else:
        bins = np.linspace(0, values.quantile(0.99), 40)

    for cls in classes:
        v = df.loc[df["class"] == cls, col]
        ax.hist(v, bins=bins, alpha=0.55, color=colors[cls], label=f"{cls} (n={len(v)})")
        ax.axvline(v.median(), color=colors[cls], linestyle="--", linewidth=1.5)
    ax.set_xlabel(label)
    ax.set_ylabel("Number of images")
    ax.legend()
fig.suptitle("Width, height and file size per class (dashed line = median)", fontsize=13)
fig.tight_layout()
fig.savefig(figdir/ "histograms.png", dpi=150)
plt.close(fig)

print(df.groupby("class")[["width", "height", "size"]].median().round(1))
square = (df["width"] == df["height"])
print("\nShare of square images per class:\n", square.groupby(df["class"]).mean().round(2))
print("\nMost common sizes per class:")
for cls in classes:
    sizes = df[df["class"] == cls].groupby(["width", "height"]).size().nlargest(3)
    print(cls, dict(sizes))

