import re
from datetime import datetime
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt


def parse_log(path: str):
    ts_pat = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    lr_pat = re.compile(r"'learning_rate'\s*:\s*([0-9.eE+-]+)")
    loss_pat = re.compile(r"'loss'\s*:\s*([0-9.eE+-]+)")
    epoch_pat = re.compile(r"'epoch'\s*:\s*([0-9.eE+-]+)")

    records = []
    pending_ts = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if ts_pat.match(line):
                pending_ts = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                continue

            m_epoch = epoch_pat.search(line)
            m_lr = lr_pat.search(line)
            m_loss = loss_pat.search(line)
            if m_epoch and m_lr and m_loss:
                records.append(
                    {
                        "epoch": float(m_epoch.group(1)),
                        "lr": float(m_lr.group(1)),
                        "loss": float(m_loss.group(1)),
                        "ts": pending_ts,
                    }
                )
                pending_ts = None

    records.sort(key=lambda r: r["epoch"])
    return records


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def group_by_discrete_epoch(records, decimals=2):
    buckets = defaultdict(list)
    for r in records:
        k = round(r["epoch"], decimals)
        buckets[k].append(r)

    out = OrderedDict()
    for k in sorted(buckets.keys()):
        out[k] = buckets[k]
    return out


def group_by_epoch_bin(records, bin_mode="int"):
    bins = defaultdict(list)
    for r in records:
        if bin_mode == "int":
            b = int(r["epoch"])
        else:
            raise ValueError("Unsupported bin_mode")
        bins[b].append(r)

    out = OrderedDict()
    for b in sorted(bins.keys()):
        out[b] = bins[b]
    return out


def compute_discrete_epoch_loss_stats(discrete_groups):
    rows = []
    for ep, recs in discrete_groups.items():
        rows.append(
            {
                "epoch": ep,
                "count": len(recs),
                "mean_loss": mean([r["loss"] for r in recs]),
            }
        )
    return rows


def compute_epoch_bin_lr_stats(epoch_bins):
    rows = []
    for b, recs in epoch_bins.items():
        rows.append(
            {
                "epoch_bin": b,
                "count": len(recs),
                "mean_lr": mean([r["lr"] for r in recs]),
                "min_lr": min(r["lr"] for r in recs),
                "max_lr": max(r["lr"] for r in recs),
            }
        )
    return rows



def plot_loss_vs_epoch(stats_rows, out_png="loss_vs_epoch.pdf"):
    xs = [r["epoch"] for r in stats_rows]
    ys = [r["mean_loss"] for r in stats_rows]

    plt.figure()
    plt.plot(xs, ys, color="royalblue", linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("mean loss")
    plt.title("Loss vs Epoch")
    plt.grid(True, linestyle="-", linewidth=0.5,color='gray', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    print(f"Saved plot to: {out_png}")


def main():
    LOG_PATH = "YOUR_LOG_PATH"
    OUT_PNG = "loss_vs_epoch.pdf"
    DECIMALS = 2

    records = parse_log(LOG_PATH)
    if not records:
        raise ValueError("没有解析到记录，检查日志格式/路径。")

    discrete_groups = group_by_discrete_epoch(records, decimals=DECIMALS)
    loss_rows = compute_discrete_epoch_loss_stats(discrete_groups)

    print("\n[Loss stats for plotting: discrete epochs (no bucketing)]")
    print("epoch\tcount\tmean_loss")
    for r in loss_rows:
        print(f"{r['epoch']:.{DECIMALS}f}\t{r['count']}\t{r['mean_loss']:.6f}")

    plot_loss_vs_epoch(loss_rows, out_png=OUT_PNG)

    epoch_bins = group_by_epoch_bin(records, bin_mode="int")
    lr_rows = compute_epoch_bin_lr_stats(epoch_bins)

    print("\n[Average LR per epoch-bin: int(epoch) bucketing]")
    print("epoch_bin\tcount\tmean_lr\t\t\tmin_lr\t\t\tmax_lr")
    for r in lr_rows:
        print(
            f"{r['epoch_bin']}\t\t{r['count']}\t{r['mean_lr']:.12e}\t"
            f"{r['min_lr']:.12e}\t{r['max_lr']:.12e}"
        )


if __name__ == "__main__":
    main()