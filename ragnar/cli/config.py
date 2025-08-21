from __future__ import annotations
import argparse
import sys
from pathlib import Path
from importlib.resources import files

from pydantic import ValidationError

from ragnar.sources.loader import validate_and_resolve


def cmd_init(args: argparse.Namespace) -> int:
    dest = Path(args.path).resolve()
    if dest.exists() and not args.force:
        print(f"[!] {dest} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1
    template = files("ragnar.templates").joinpath("sources.example.yaml")
    content = template.read_text(encoding="utf-8")
    dest.write_text(content, encoding="utf-8")
    print(f"[ok] Wrote template to {dest}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg_path = Path(args.file).resolve()
    try:
        resolved = validate_and_resolve(cfg_path)
    except (FileNotFoundError, ValidationError, ValueError) as e:
        print(f"[!] Invalid config: {e}", file=sys.stderr)
        return 2

    print(f"[ok] Config valid: {cfg_path}")
    for name, cfg in resolved.items():
        kind = getattr(cfg, "kind", "?")
        if kind == "markdown_repo":
            print(f"  - {name} ({kind})")
            print(f"      repo_path: {cfg.repo_path}")
            print(f"      base_url:  {cfg.base_url}")
            print(f"      include:   {cfg.include_globs}")
            print(f"      exclude:   {cfg.exclude_dirs}")
            print(f"      lang:      {cfg.default_lang}")
        else:
            print(f"  - {name} ({kind})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragnar-config", description="RAGnaR config utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a sources.yaml from the packaged template")
    p_init.add_argument("-p", "--path", default="sources.yaml", help="Destination path")
    p_init.add_argument("-f", "--force", action="store_true", help="Overwrite if exists")
    p_init.set_defaults(func=cmd_init)

    p_val = sub.add_parser("validate", help="Validate and show resolved sources")
    p_val.add_argument("-f", "--file", default="sources.yaml", help="Path to sources.yaml")
    p_val.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    raise SystemExit(rc)
