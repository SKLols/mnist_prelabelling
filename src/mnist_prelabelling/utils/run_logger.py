import json
from datetime import datetime
from pathlib import Path


def save_run_log(run_name: str, data: dict, output_dir: str = "outputs/runs") -> str:
    """Save a dictionary of run metadata/results to a timestamped JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_name}_{timestamp}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Run log saved to {filepath}")
    return str(filepath)