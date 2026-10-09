"""Download the pretrained models DataGuardian uses into backend/models/ (one-time, ~90 MB).

    .venv/bin/python -m scripts.download_models

After this, scans run fully offline. Isolation Forest needs no download: it is fitted on each
dataset during the scan. RapidFuzz is a library, not a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402

REPO = "sentence-transformers/all-MiniLM-L6-v2"
# Only what the PyTorch path needs (skip ONNX / OpenVINO / TF variants in the repo).
ALLOW = [
    "config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    "model.safetensors", "1_Pooling/config.json", "README.md",
]


def main() -> None:
    from huggingface_hub import snapshot_download

    target = get_settings().models_dir / "all-MiniLM-L6-v2"
    target.mkdir(parents=True, exist_ok=True)
    try:  # reuse the Hugging Face cache if this machine already has the model
        cached = Path(snapshot_download(repo_id=REPO, allow_patterns=ALLOW, local_files_only=True))
        if not (cached / "model.safetensors").exists():
            raise FileNotFoundError
        import shutil

        print(f"Copying {REPO} from local Hugging Face cache {cached}")
        shutil.copytree(cached, target, dirs_exist_ok=True)
    except Exception:
        print(f"Downloading {REPO} -> {target}")
        snapshot_download(repo_id=REPO, local_dir=str(target), allow_patterns=ALLOW)
    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"Done: {size / 1024**2:.1f} MB")

    from app.ml.embeddings import SemanticModel

    m = SemanticModel(target)
    sim = m.similarity("customer mobile number", "phone")
    print(f"Self-test OK · cosine('customer mobile number', 'phone') = {sim:.3f}")


if __name__ == "__main__":
    main()
