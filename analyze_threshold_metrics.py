import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-9


def summarize_by_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std of dice/iou by threshold."""
    agg = (df.groupby("threshold")
             .agg(
                 dice_mean=("dice", "mean"),
                 dice_median=("dice", "median"),
                 dice_std=("dice", "std"),
                 iou_mean=("iou", "mean"),
                 iou_median=("iou", "median"),
                 iou_std=("iou", "std"),
                 n=("dice", "count"),
             )
             .reset_index()
             .sort_values("threshold"))
    return agg


def _ensure_endpoints_roc(fpr: np.ndarray, tpr: np.ndarray):
    """
    Add ROC endpoints (0,0) and (1,1) if missing and sort by fpr.
    """
    pts = np.column_stack([fpr, tpr])
    pts = pts[~np.isnan(pts).any(axis=1)]
    pts = pts[np.argsort(pts[:, 0])]

    # prepend (0,0) if needed
    if len(pts) == 0 or (pts[0, 0] > 0 + 1e-12 or pts[0, 1] > 0 + 1e-12):
        pts = np.vstack([[0.0, 0.0], pts])

    # append (1,1) if needed
    if pts[-1, 0] < 1 - 1e-12 or pts[-1, 1] < 1 - 1e-12:
        pts = np.vstack([pts, [1.0, 1.0]])

    return pts[:, 0], pts[:, 1]


def _ensure_endpoints_pr(recall: np.ndarray, precision: np.ndarray):
    """
    For PR curves, add a sensible endpoint at recall=0 with precision=1.
    Then sort by recall.
    """
    pts = np.column_stack([recall, precision])
    pts = pts[~np.isnan(pts).any(axis=1)]
    pts = pts[np.argsort(pts[:, 0])]

    if len(pts) == 0 or pts[0, 0] > 0 + 1e-12:
        pts = np.vstack([[0.0, 1.0], pts])

    # No strict need to force recall=1 endpoint; depends on thresholds.
    # But adding it can help interpretability if your highest-recall point isn't 1.
    # We'll add recall=1 with the last known precision (or 0 if none).
    if pts[-1, 0] < 1 - 1e-12:
        pts = np.vstack([pts, [1.0, pts[-1, 1]]])

    return pts[:, 0], pts[:, 1]


def compute_pixelwise_auc_global(df: pd.DataFrame):
    """
    Compute global (micro) pixelwise ROC AUC and PR AUC using summed TP/FP/FN/TN across all images
    for each threshold.
    Returns dict with curves + AUCs.
    """
    req = {"tp", "fp", "fn", "tn"}
    if not req.issubset(df.columns):
        return None

    g = (df.groupby("threshold")[["tp", "fp", "fn", "tn"]]
           .sum()
           .reset_index()
           .sort_values("threshold"))

    tp = g["tp"].to_numpy(dtype=float)
    fp = g["fp"].to_numpy(dtype=float)
    fn = g["fn"].to_numpy(dtype=float)
    tn = g["tn"].to_numpy(dtype=float)

    tpr = tp / (tp + fn + EPS)
    fpr = fp / (fp + tn + EPS)

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    fpr_s, tpr_s = _ensure_endpoints_roc(fpr, tpr)
    roc_auc = float(np.trapezoid(tpr_s, fpr_s))

    rec_s, prec_s = _ensure_endpoints_pr(recall, precision)
    pr_auc = float(np.trapezoid(prec_s, rec_s))

    return {
        "thresholds": g["threshold"].to_numpy(),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr_sorted": fpr_s,
        "tpr_sorted": tpr_s,
        "recall_sorted": rec_s,
        "precision_sorted": prec_s,
        "counts_by_threshold": g,
    }


def compute_pixelwise_auc_per_image(df: pd.DataFrame):
    """
    Compute per-image ROC AUC and PR AUC by building ROC/PR curves across thresholds
    from that image's TP/FP/FN/TN at each threshold.
    Returns a DataFrame with one row per sample_id.
    """
    req = {"tp", "fp", "fn", "tn", "sample_id"}
    if not req.issubset(df.columns):
        return None

    out_rows = []
    for sample_id, sub in df.groupby("sample_id"):
        sub = sub.sort_values("threshold")
        tp = sub["tp"].to_numpy(dtype=float)
        fp = sub["fp"].to_numpy(dtype=float)
        fn = sub["fn"].to_numpy(dtype=float)
        tn = sub["tn"].to_numpy(dtype=float)

        tpr = tp / (tp + fn + EPS)
        fpr = fp / (fp + tn + EPS)

        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)

        fpr_s, tpr_s = _ensure_endpoints_roc(fpr, tpr)
        roc_auc = float(np.trapezoid(tpr_s, fpr_s))

        rec_s, prec_s = _ensure_endpoints_pr(recall, precision)
        pr_auc = float(np.trapezoid(prec_s, rec_s))

        out_rows.append({
            "sample_id": sample_id,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        })

    return pd.DataFrame(out_rows)


def save_lineplot(x, y_series: dict, xlabel, ylabel, title, out_path: Path, show: bool):
    plt.figure()
    for label, y in y_series.items():
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def save_scatter(x, y, xlabel, ylabel, title, out_path: Path, show: bool):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metrics_out/per_image_threshold_metrics.csv")
    ap.add_argument("--out_dir", default="metrics_out")
    ap.add_argument("--show", action="store_true", help="Show plots interactively (otherwise just save)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # ---- Basic summary (what you had before) ----
    agg = summarize_by_threshold(df)
    summary_csv = out_dir / "threshold_summary_stats.csv"
    agg.to_csv(summary_csv, index=False)

    print("\nSummary by threshold (Dice/IoU):")
    print(agg)
    print(f"\nSaved summary CSV: {summary_csv}")

    # Mean/median plots
    save_lineplot(
        agg["threshold"],
        {"Dice mean": agg["dice_mean"], "Dice median": agg["dice_median"]},
        "Threshold", "Dice", "Dice vs Threshold",
        out_dir / "dice_vs_threshold.png",
        show=args.show
    )

    save_lineplot(
        agg["threshold"],
        {"IoU mean": agg["iou_mean"], "IoU median": agg["iou_median"]},
        "Threshold", "IoU", "IoU vs Threshold",
        out_dir / "iou_vs_threshold.png",
        show=args.show
    )

    # Std plot
    save_lineplot(
        agg["threshold"],
        {"Dice std": agg["dice_std"], "IoU std": agg["iou_std"]},
        "Threshold", "Std Dev", "Std Dev vs Threshold",
        out_dir / "std_vs_threshold.png",
        show=args.show
    )

    # ---- Scatter plots per threshold: every image's Dice/IoU ----
    scat_dir = out_dir / "scatter_by_threshold"
    scat_dir.mkdir(exist_ok=True)

    # Use image index within threshold group for x-axis (clean + stable)
    thresholds = sorted(df["threshold"].unique())
    for thr in thresholds:
        sub = df[df["threshold"] == thr].copy()
        sub = sub.sort_values("sample_id") if "sample_id" in sub.columns else sub.reset_index(drop=True)
        x = np.arange(len(sub))

        save_scatter(
            x, sub["dice"].to_numpy(),
            "Image # (sorted)", "Dice",
            f"Dice scatter @ threshold={thr:.2f}",
            scat_dir / f"dice_scatter_thr_{thr:.2f}.png",
            show=args.show
        )

        save_scatter(
            x, sub["iou"].to_numpy(),
            "Image # (sorted)", "IoU",
            f"IoU scatter @ threshold={thr:.2f}",
            scat_dir / f"iou_scatter_thr_{thr:.2f}.png",
            show=args.show
        )

        # Dice vs IoU scatter (per image, same threshold)
        save_scatter(
            sub["dice"].to_numpy(), sub["iou"].to_numpy(),
            "Dice", "IoU",
            f"Dice vs IoU @ threshold={thr:.2f}",
            scat_dir / f"dice_vs_iou_thr_{thr:.2f}.png",
            show=args.show
        )

    print(f"\nSaved scatter plots to: {scat_dir}")

    # ---- Pixelwise AUC (needs tp/fp/fn/tn) ----
    global_auc = compute_pixelwise_auc_global(df)
    per_img_auc = compute_pixelwise_auc_per_image(df)

    if global_auc is None or per_img_auc is None:
        print("\n[WARN] Pixelwise AUC requested, but your CSV is missing tp/fp/fn/tn (and/or sample_id).")
        print("       Add columns: tp, fp, fn, tn per image per threshold in your collector CSV, then rerun this analyzer.")
        return

    # Global (micro) ROC/PR curves
    print("\nPixelwise AUC (GLOBAL / micro):")
    print(f"  ROC AUC = {global_auc['roc_auc']:.6f}")
    print(f"  PR  AUC = {global_auc['pr_auc']:.6f}")

    # Save global AUC summary
    auc_summary_path = out_dir / "pixelwise_auc_global.txt"
    auc_summary_path.write_text(
        f"Pixelwise AUC (GLOBAL / micro)\nROC AUC: {global_auc['roc_auc']:.8f}\nPR AUC: {global_auc['pr_auc']:.8f}\n"
    )
    print(f"Saved: {auc_summary_path}")

    # Plot global ROC
    plt.figure()
    plt.plot(global_auc["fpr_sorted"], global_auc["tpr_sorted"], marker="o")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pixelwise ROC (global)  AUC={global_auc['roc_auc']:.4f}")
    roc_path = out_dir / "pixelwise_roc_global.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close()

    # Plot global PR
    plt.figure()
    plt.plot(global_auc["recall_sorted"], global_auc["precision_sorted"], marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Pixelwise PR (global)  AUC={global_auc['pr_auc']:.4f}")
    pr_path = out_dir / "pixelwise_pr_global.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"Saved global ROC/PR plots:\n- {roc_path}\n- {pr_path}")

    # Per-image AUC stats
    def stats(series: pd.Series):
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }

    roc_stats = stats(per_img_auc["roc_auc"])
    pr_stats = stats(per_img_auc["pr_auc"])

    print("\nPixelwise AUC (PER-IMAGE) summary:")
    print(f"  ROC AUC  mean={roc_stats['mean']:.6f}  median={roc_stats['median']:.6f}  std={roc_stats['std']:.6f}")
    print(f"  PR  AUC  mean={pr_stats['mean']:.6f}  median={pr_stats['median']:.6f}  std={pr_stats['std']:.6f}")

    per_img_auc_csv = out_dir / "pixelwise_auc_per_image.csv"
    per_img_auc.to_csv(per_img_auc_csv, index=False)
    print(f"Saved per-image AUC CSV: {per_img_auc_csv}")

    # Optional: histogram-ish scatter (AUC vs image index)
    x = np.arange(len(per_img_auc))
    save_scatter(
        x, per_img_auc["roc_auc"].to_numpy(),
        "Image #", "ROC AUC",
        "Per-image pixelwise ROC AUC",
        out_dir / "pixelwise_roc_auc_per_image_scatter.png",
        show=args.show
    )
    save_scatter(
        x, per_img_auc["pr_auc"].to_numpy(),
        "Image #", "PR AUC",
        "Per-image pixelwise PR AUC",
        out_dir / "pixelwise_pr_auc_per_image_scatter.png",
        show=args.show
    )


if __name__ == "__main__":
    main()
