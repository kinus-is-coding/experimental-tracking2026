# 📊 Model Evaluation Guide

This repository is used by trainers to evaluate model performance after training.

It generates:

Confusion matrix
Error analysis (what the model predicts wrong)
Per-writer performance
learning curves
# ⚙️ 1. Setup

Install dependencies:

    pip install pandas numpy matplotlib seaborn scikit-learn pillow
# 📂 2. Required Files

After training, make sure you have:

✅ predictions.csv 
y_true,y_pred,writer,image_path


✅ history.csv 
epoch,train_loss,test_loss,train_acc,test_acc
# 🚀 3. Run Evaluation

With learning curves
    python evaluate.py \
    --input predictions.csv \ 
    --history history.csv \
    --output runs/[REGIME_NAME]/[RUN_ID]_[DATASET]

🔑 Components
REGIME

Training setup:

REGIME1 → MNIST training
REGIME2 → Single Writer
REGIME3 → Mixed Writers
REGIME4 → All Writers
RUN_ID

Experiment ID:

exp01, exp02, exp03, ...
DATASET

Evaluation dataset:

mnist
custom
mnist
custom 
#📁 4. Output

After running, results are saved in:

runs/REGIMEX/.../
Generated files:
confusion_matrix.png → overall prediction results
confusion_matrix_normalized.png → normalized version
error_log.csv → all wrong predictions
error_stats.json → most common mistakes
error_samples.png → images the model got wrong (if image_path exists)
writers/*.png → per-writer performance (if writer exists)
curves.png → training curves (if history.csv provided)
# ⚠️ 5. Common Issues
No error images

→ image_path missing or incorrect

No curves

→ history.csv not provided

Script error

→ check required columns:

y_true, y_pred
✔️ Workflow
Train model
   ↓
Export predictions.csv (+ history.csv)
   ↓
Run evaluate.py
   ↓
Check outputs
✔️ Done

PLEASE REMEMBER TO PUSH TO GITHUB!


