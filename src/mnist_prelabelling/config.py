from datetime import datetime
from pathlib import Path

# data
DATA_ROOT = "data"

# reproducibility
RANDOM_SEED = 42

# model selection
MODEL_NAME = "custom_cnn"   # or "lenet5" — read by the factory

# shared training hyperparameters (same for both models — fair comparison)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# shared across all models
NUM_CLASSES = 10

# seed-set training
VAL_FRACTION = 0.1

def get_run_output_dir() -> Path:
    """Create and return a timestamped output directory for one pipeline run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "runs").mkdir(exist_ok=True)
    return run_dir