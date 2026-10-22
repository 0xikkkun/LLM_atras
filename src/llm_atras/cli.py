"""Portable CLI for the 15-category research protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

CATEGORIES = ["mathematics", "programming", "education", "computer science",
              "medicine", "psychology", "philosophy", "technology", "physics",
              "biology", "economics", "politics", "business", "literature", "engineering"]


def fetch_texts(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    ds = load_dataset(args.dataset, split="train", revision=revision, streaming=True)
    ds = ds.shuffle(seed=20261022, buffer_size=1000)
    groups = {c: [] for c in CATEGORIES}
    seen = set()
    for index, row in enumerate(ds):
        text = (row.get("full_text") or "").strip()
        category = (row.get("category") or "").lower().strip()
        if category in groups and len(groups[category]) < 20 and len(text) > 50 and text not in seen:
            groups[category].append(text)
            seen.add(text)
        if all(len(v) == 20 for v in groups.values()):
            break
        if index >= 100000:
            raise RuntimeError("Could not collect 20 unique texts in every category")
    if not all(len(v) == 20 for v in groups.values()):
        raise RuntimeError("Dataset exhausted before all category quotas were met")
    result = dict(texts=[t for c in CATEGORIES for t in groups[c]],
                  labels=[c for c in CATEGORIES for t in groups[c]],
                  dataset=args.dataset, revision=revision, seed=20261022)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.source.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved 300 texts to {args.source}; dataset revision {revision}")


def run_stage(args):
    models = json.loads((args.output / "models.json").read_text())
    config = json.loads((args.output / "config.json").read_text())
    indices = config["smoke_models"] if args.stage == "smoke" else list(range(len(models)))
    gpus = [s.strip() for s in args.gpus.split(",") if s.strip()]
    if not gpus:
        raise ValueError("Specify at least one GPU with --gpus")
    logs = args.output / args.stage / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def worker(gpu, batch):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4")
        for index in batch:
            command = [sys.executable, "-m", "llm_atras", "extract", "--output", str(args.output),
                       "--stage", args.stage, "--index", str(index)]
            with (logs / f"extract_{index:02d}.log").open("a") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        tasks = [pool.submit(worker, gpu, indices[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        for task in tasks:
            task.result()
    command = [sys.executable, "-m", "llm_atras", "analyze", "--output", str(args.output), "--stage", args.stage]
    if args.quick:
        command.append("--quick")
    subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "extract", "analyze", "run", "selftest"]:
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, default=Path("artifacts"))
        if command in ["extract", "analyze", "run"]:
            p.add_argument("--stage", choices=["smoke", "main"], default="smoke")
        if command == "prepare":
            p.add_argument("--source", type=Path, required=True)
            p.add_argument("--config", type=Path, help="JSON containing models and smoke_models")
            p.add_argument("--cache-dir", type=Path, action="append", help="HF hub cache; repeat for multiple caches")
        if command == "extract":
            p.add_argument("--index", type=int, required=True)
        if command in ["analyze", "run"]:
            p.add_argument("--quick", action="store_true", help="Skip approximate Gromov-Wasserstein")
        if command == "run":
            p.add_argument("--gpus", default="0", help="Comma-separated visible GPU indices")
    p = sub.add_parser("fetch-texts")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dataset", default="ykwon-hf/unconditional_text_generation")
    p.add_argument("--revision")
    args = parser.parse_args(argv)
    if args.command == "fetch-texts":
        fetch_texts(args)
        return
    args.output = args.output.resolve()
    if args.command == "run":
        run_stage(args)
        return
    from . import pipeline
    pipeline.ROOT = args.output
    if args.command == "prepare":
        pipeline.SOURCE = args.source.resolve()
        if args.cache_dir:
            pipeline.CACHES = [p.expanduser().resolve() for p in args.cache_dir]
        if args.config:
            config = json.loads(args.config.read_text())
            pipeline.MODELS = [(m["id"], m["family"]) for m in config["models"]]
            pipeline.SMOKE = config.get("smoke_models", list(range(min(3, len(pipeline.MODELS)))))
        if len(pipeline.MODELS) < 3 or len(pipeline.SMOKE) < 3:
            parser.error("At least three models are required in both stages")
        if len(set(m[0] for m in pipeline.MODELS)) != len(pipeline.MODELS):
            parser.error("Model IDs must be unique")
        if any(i < 0 or i >= len(pipeline.MODELS) for i in pipeline.SMOKE):
            parser.error("smoke_models contains an invalid index")
        pipeline.prepare()
    elif args.command == "extract":
        pipeline.extract(args.stage, args.index)
    elif args.command == "analyze":
        pipeline.analyze(args.stage, args.quick)
    else:
        pipeline.selftest()
