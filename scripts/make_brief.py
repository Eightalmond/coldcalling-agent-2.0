"""Build a product brief from your own material, for the live-call bot.

Examples:
  uv run python scripts/make_brief.py --name Quillo --company "Brightfern Labs" \\
      --description "AI meeting notes" --file examples/quillo_material.md
  uv run python scripts/make_brief.py --name Acme --company "Acme Inc" --text "..." --web

Then run the bot with it:
  uv run python -m calle.bot --brief briefs/quillo.md

Prints warnings, the output path and the brief's size. Never prints keys,
the material, or the brief itself (open the file to review it).
"""

import argparse
import sys
from pathlib import Path

from calle.config import ConfigError, load_settings
from calle.knowledge import (
    KnowledgeError,
    ProductInput,
    build_brief,
    load_material,
    slugify,
)

BRIEFS_DIR = Path("briefs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Condense product material into a brief.")
    parser.add_argument("--name", required=True, help="product name")
    parser.add_argument("--company", required=True, help="company name")
    parser.add_argument("--description", default="", help="one-line product description")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="material file: .txt, .md or .pdf")
    source.add_argument("--text", help="material as pasted text")
    parser.add_argument("--web", action="store_true", help="add Tavily web results (needs TAVILY_API_KEY)")
    parser.add_argument("--out", type=Path, help="output path (default: briefs/<product-slug>.md)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings()
        material = load_material(args.file) if args.file else args.text
        product = ProductInput(
            product_name=args.name,
            company_name=args.company,
            description=args.description,
            material_text=material,
            use_web=args.web,
        )
        print(f"Building brief for {args.name} ({len(material):,} chars of material)...")
        result = build_brief(product, settings)
    except (ConfigError, KnowledgeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for warning in result.warnings:
        print(f"Warning: {warning}")

    out = args.out or BRIEFS_DIR / f"{slugify(args.name)}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.brief + "\n", encoding="utf-8")
    web = "with web context" if result.used_web else "material only"
    print(f"Wrote {out} ({len(result.brief):,} chars, {web}, attempt {result.attempts}).")
    print(f"Review it, then run:  uv run python -m calle.bot --brief {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
