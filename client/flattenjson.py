import os
import json
import numpy as np
from pathlib import Path

# Directory to process (current directory)
dir_path = Path(".")

for file in dir_path.glob("*.json"):
    try:
        with open(file, "r") as f:
            data = json.load(f)

        if "itls" in data and isinstance(data["itls"], list):
            data["itls"] = [float(np.mean(x)) for x in data["itls"] if isinstance(x, list)]

            # Write to backup
            backup_path = file.with_suffix(file.suffix + ".bak")
            with open(backup_path, "w") as backup:
                json.dump(data, backup, indent=2)

            # Overwrite original safely after successful write
            os.replace(backup_path, file)

            print(f"Processed: {file}")
        else:
            print(f"Skipped (no 'itls'): {file}")

    except Exception as e:
        print(f"Error processing {file}: {e}")

