from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import load_protocol, load_settings, prepare


def main():
    parser = argparse.ArgumentParser(description="Isolated multilingual attention-group donor ablations.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        sub = commands.add_parser(command, help="Freeze question and donor choices" if command == "prepare" else "Run patching and analysis (resumable)")
        sub.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/qwen_belebele.yaml")
        sub.add_argument("--samples-per-language", type=int, help="Number of aligned recipient questions (default: 100 in supplied configs)")
        sub.add_argument("--output-dir", type=Path, help="New run directory; relative to the current working directory")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a setting using YAML values")
        if command == "run":
            sub.add_argument("--skip-analysis", action="store_true")
    sub = commands.add_parser("analyze", help="Analyze an already completed run; no model is loaded")
    sub.add_argument("--run-dir", type=Path, required=True)
    sub.add_argument("--bootstrap-resamples", type=int)
    args = parser.parse_args()
    if args.command == "analyze":
        from .analysis import analyze
        analyze(args.run_dir.resolve(), args.bootstrap_resamples)
        print(f"Analysis saved in {args.run_dir.resolve() / 'analysis'}")
        return
    cfg = load_settings(args.config, args.set, args.samples_per_language, args.output_dir)
    output = Path(cfg["run"]["output_dir"])
    protocol_path = output / "protocol.json"
    protocol = load_protocol(protocol_path, cfg) if protocol_path.exists() else prepare(cfg, protocol_path)
    n = len(protocol["recipient_ids"])
    print(f"Protocol: {protocol_path}\nRecipients: {n} questions x {len(protocol['languages'])} languages; "
          f"{len(protocol['expanded_samples'])} prompts including answer orders.\n"
          f"Reference donor pool: {len(protocol['reference_ids'])} questions per language.\n"
          f"Variants: {', '.join(v['name'] for v in protocol['variants'])}", flush=True)
    if args.command == "run":
        from .engine import run
        run(protocol, output)
        if not args.skip_analysis:
            from .analysis import analyze
            analyze(output)
        print(f"Completed: {output}")


if __name__ == "__main__":
    main()
