"""CLI for raw-review inference with a versioned sentiment artifact."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment from raw Vietnamese reviews.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--text", nargs="+", required=True, help="One or more quoted raw reviews.")
    parser.add_argument("--probabilities", action="store_true", help="Include uncalibrated model probabilities.")
    args = parser.parse_args()
    model = load_model(args.model_path)
    print(json.dumps({"artifact_version": model.artifact_version,
        "predictions": model.infer(args.text, include_probabilities=args.probabilities)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
