#!/usr/bin/env python3
import csv, math
from collections import defaultdict

def mn(v):
    return sum(v) / len(v)

def sd(v):
    m = mn(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

# Part 1
with open("/home/zeyufu/Desktop/MoCoO/benchmarks/results/multiseed/multiseed_IRALL.csv") as f:
    rows = list(csv.DictReader(f))

d = defaultdict(lambda: defaultdict(list))
for r in rows:
    for c in ["ARI", "NMI", "ASW", "CH", "DB", "train_time_s"]:
        d[r["config"]][c].append(float(r[c]))

print("=== MULTISEED IRALL ===")
for cfg in ["VAE", "VAE+ODE", "VAE+MoCo", "VAE+MoCo+Proto", "VAE+ODE+MoCo", "Full"]:
    parts = []
    for c in ["ARI", "NMI", "ASW", "CH", "DB", "train_time_s"]:
        m = mn(d[cfg][c])
        s = sd(d[cfg][c])
        if c in ("CH", "train_time_s"):
            parts.append("%s: %.1f +/- %.1f" % (c, m, s))
        else:
            parts.append("%s: %.3f +/- %.3f" % (c, m, s))
    print(cfg)
    for p in parts:
        print("  " + p)

# Part 2
print("")
print("=== PAUL BETA SWEEP ===")
bc = ["full_ARI", "full_NMI", "full_ASW", "full_CH", "full_DB"]
for bv in ["0.01", "0.1", "1.0"]:
    path = "/home/zeyufu/Desktop/MoCoO/benchmarks/results/paul_beta_ablation/beta_%s/summary.csv" % bv
    with open(path) as f:
        br = list(csv.DictReader(f))
    print("")
    print("beta = %s" % bv)
    for row in br:
        parts = []
        for c in bc:
            v = float(row[c])
            if c == "full_CH":
                parts.append("%s: %.1f" % (c, v))
            else:
                parts.append("%s: %.3f" % (c, v))
        print("  %s: %s" % (row["config"], " | ".join(parts)))
