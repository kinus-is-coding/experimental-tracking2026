import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from PIL import Image

# =========================
# MAIN EVALUATION
# =========================
def evaluate(pred_csv, output_dir="evaluation_output", max_error_images=20):
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/plots/writers", exist_ok=True)
    os.makedirs(f"{output_dir}/errors", exist_ok=True)

    df = pd.read_csv(pred_csv)

    required_cols = ["y_true", "y_pred"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    y_true = df["y_true"].values
    y_pred = df["y_pred"].values

    # =========================
    # 1. GLOBAL CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{output_dir}/plots/confusion_matrix.png", dpi=300)
    plt.close()

    # =========================
    # 2. NORMALIZED CM
    # =========================
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Confusion Matrix (Normalized)")
    plt.savefig(f"{output_dir}/plots/confusion_matrix_normalized.png", dpi=300)
    plt.close()

    # =========================
    # 3. PER-WRITER CM
    # =========================
    if "writer" in df.columns:
        for w in df["writer"].unique():
            sub = df[df["writer"] == w]
            cm_w = confusion_matrix(sub["y_true"], sub["y_pred"])

            plt.figure(figsize=(5,4))
            sns.heatmap(cm_w, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Writer {w}")
            plt.savefig(f"{output_dir}/plots/writers/writer_{w}.png", dpi=300)
            plt.close()

    # =========================
    # 4. ERROR LOG
    # =========================
    error_df = df[df["y_true"] != df["y_pred"]]
    error_df.to_csv(f"{output_dir}/error_log.csv", index=False)

    # =========================
    # 5. ERROR STATISTICS
    # =========================
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    counter = Counter(pairs)

    top_errors = {
        f"{k[0]}->{k[1]}": v
        for k, v in counter.most_common(10)
    }

    with open(f"{output_dir}/error_stats.json", "w") as f:
        json.dump(top_errors, f, indent=4)

    # =========================
    # 6. ERROR IMAGE GRID
    # =========================
    if "image_path" in df.columns:
        error_samples = error_df.head(max_error_images)

        plt.figure(figsize=(10,8))

        for i, (_, row) in enumerate(error_samples.iterrows()):
            try:
                img = Image.open(row["image_path"]).convert("L")

                plt.subplot(4,5,i+1)
                plt.imshow(img, cmap="gray")
                plt.title(f"{row['y_true']}→{row['y_pred']}")
                plt.axis("off")
            except:
                continue

        plt.suptitle("Misclassified Samples")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/errors/error_samples.png", dpi=300)
        plt.close()

    print("✅ Evaluation complete.")
    print(f"📁 Output: {output_dir}")



def plot_learning_curves(history_csv, output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(history_csv)

    required_cols = ['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc']
    for col in required_cols:
        if col not in df.columns:
            print(f"[WARN] Missing column {col}, skip curves")
            return

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # LOSS
    ax1.plot(df['epoch'], df['train_loss'], '--', label='Train Loss')
    ax1.plot(df['epoch'], df['test_loss'], label='Test Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ACCURACY
    ax2.plot(df['epoch'], df['train_acc'], '--', label='Train Acc')
    ax2.plot(df['epoch'], df['test_acc'], label='Test Acc')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/curves.png", dpi=300)
    plt.close()

    print("📈 Saved curves.png")
# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to predictions.csv")
    parser.add_argument("--history", default=None, help="Path to history.csv")
    parser.add_argument("--output", default="evaluation_output")

    args = parser.parse_args()

    evaluate(args.input, args.output)
    plot_learning_curves(args.history, args.output)

