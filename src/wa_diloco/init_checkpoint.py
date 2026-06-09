from __future__ import annotations

import argparse
from pathlib import Path

from wa_diloco.config import ensure_workspace, load_config
from wa_diloco.state import save_state
from wa_diloco.telemetry import append_jsonl


def run(config_path: str | Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    ensure_workspace(config)

    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name_or_path)

    checkpoint = config.runtime.workspace / "checkpoints" / "global_round_0000.pt"
    save_state(checkpoint, model.state_dict())
    tokenizer.save_pretrained(config.runtime.workspace / "tokenizer")

    append_jsonl(
        config.runtime.workspace / "events" / "init.jsonl",
        {
            "event": "init_checkpoint",
            "model": config.model.name_or_path,
            "checkpoint": str(checkpoint),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

