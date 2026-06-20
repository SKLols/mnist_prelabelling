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