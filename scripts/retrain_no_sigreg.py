"""
Retrain S-JEPA with SIGReg disabled (lambda=0).
Saves to checkpoints_no_sigreg/ — does NOT touch your main checkpoints.
Runtime: ~26 min (same as normal training).

Run:
    python retrain_no_sigreg.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.train_jepa import train_jepa

print("Retraining S-JEPA with SIGReg disabled (sigreg_lambda=0)...")
print("Saving to: checkpoints_no_sigreg/")
print("This will NOT overwrite your main checkpoints.\n")

train_jepa(
    sigreg_lambda=0,        # SIGReg OFF
    epochs=28,
    save_dir="checkpoints_no_sigreg",
)

# Copy the final epoch to a named checkpoint
import shutil
src = Path("checkpoints_no_sigreg/game_jepa_ep28.pth")
dst = Path("checkpoints/game_jepa_no_sigreg.pth")
shutil.copy(src, dst)
print(f"\nCopied {src} -> {dst}")
print("\nNow run the Unity ablation:")
print("  python run_ablations_live.py --level 1 --no-sigreg")
print("  python run_ablations_live.py --level 2 --no-sigreg")
print("  python run_ablations_live.py --level 3 --no-sigreg")
print("  python run_ablations_live.py --level 4 --no-sigreg")
print("  python run_ablations_live.py --report")
