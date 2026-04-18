import matplotlib.pyplot as plt
import pandas as pd
import os

def save_learning_curves(history_csv, run_id, output_dir):
    """
    Hàm chuẩn để anh em gọi sau khi train xong.
    """
    df = pd.read_csv(history_csv)
    
    # Style chuẩn Academic cho báo cáo
    plt.style.use('seaborn-v0_8-paper') # Hoặc 'ggplot'
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(df['epoch'], df['train_loss'], '--', label='Train Loss', color='#1f77b4')
    ax1.plot(df['epoch'], df['test_loss'], label='Test Loss', color='#ff7f0e')
    ax1.set_title(f'Loss Curve - {run_id}')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(df['epoch'], df['train_acc'], '--', label='Train Acc', color='#2ca02c')
    ax2.plot(df['epoch'], df['test_acc'], label='Test Acc', color='#d62728')
    ax2.set_title(f'Accuracy Curve - {run_id}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}_curves.png"), dpi=300)
    print(f">>> [SYSTEM] Saved curves to {output_dir}")