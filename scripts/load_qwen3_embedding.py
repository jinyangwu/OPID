import argparse
import importlib.util
from pathlib import Path


DEFAULT_MODEL_PATH = "/raid3/data/GTPO/MODELS/Qwen3-Embedding-0.6B"


# def apply_local_compat_patches():
#     repo_root = Path(__file__).resolve().parents[1]
#     sitecustomize_path = repo_root / "external_source" / "sitecustomize.py"
#     if not sitecustomize_path.exists():
#         return

#     spec = importlib.util.spec_from_file_location(
#         "_skillrl_external_sitecustomize",
#         sitecustomize_path,
#     )
#     if spec is None or spec.loader is None:
#         return
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load Qwen3-Embedding-0.6B with sentence-transformers and run a small smoke test."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local embedding model path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. cuda:0 or cpu. Defaults to sentence-transformers auto device.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text to encode. May be passed multiple times.",
    )
    return parser.parse_args()


def cosine_similarity(left, right):
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(a) * float(a) for a in left) ** 0.5
    right_norm = sum(float(b) * float(b) for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def main():
    # apply_local_compat_patches()
    args = parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: sentence_transformers. Install it in this environment, "
            "then rerun this script."
        ) from exc

    model_kwargs = {}
    if args.device:
        model_kwargs["device"] = args.device

    model = SentenceTransformer(args.model_path, **model_kwargs)
    texts = args.text or [
        "find a red mug under 20 dollars",
        "search for a cheap red cup",
        "clean the kitchen sink",
    ]
    embeddings = model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    device = getattr(model, "device", "unknown")
    print(f"model_path: {args.model_path}")
    print(f"device: {device}")
    print(f"num_texts: {len(texts)}")
    print(f"embedding_shape: {embeddings.shape}")
    for idx, text in enumerate(texts):
        print(f"text[{idx}]: {text}")
    if len(texts) >= 2:
        sim = cosine_similarity(embeddings[0], embeddings[1])
        print(f"cosine(text[0], text[1]): {sim:.6f}")


if __name__ == "__main__":
    main()
