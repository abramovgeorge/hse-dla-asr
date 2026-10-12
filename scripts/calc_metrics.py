import argparse
from pathlib import Path

from src.metrics.utils import edit_distance
from src.text_encoder import CTCTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--predicted-dir", type=Path, required=True)
    args = parser.parse_args()

    char_errors = 0
    word_errors = 0
    total_chars = 0
    total_words = 0

    target_paths = []
    for path in args.target_dir.iterdir():
        if path.is_file() and path.suffix == ".txt":
            target_paths.append(path)

    if len(target_paths) == 0:
        raise ValueError("No target .txt files found")

    for target_path in sorted(target_paths):
        predicted_path = args.predicted_dir / target_path.name
        if not predicted_path.is_file():
            raise FileNotFoundError(f"Missing prediction: {predicted_path}")

        with target_path.open(encoding="utf-8") as f:
            target_text = f.read().strip()
        with predicted_path.open(encoding="utf-8") as f:
            predicted_text = f.read().strip()

        target_text = CTCTextEncoder.normalize_text(target_text)
        predicted_text = CTCTextEncoder.normalize_text(predicted_text)

        char_errors += edit_distance(target_text, predicted_text)
        word_errors += edit_distance(target_text.split(), predicted_text.split())

        total_chars += len(target_text)
        total_words += len(target_text.split())

    if total_chars == 0 or total_words == 0:
        raise ValueError("Metrics are undefined due to target being empty")
        return

    print(f"CER is {char_errors / total_chars}, WER is {word_errors / total_words}")


if __name__ == "__main__":
    main()
