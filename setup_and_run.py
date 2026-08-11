#!/usr/bin/env python
"""
setup_and_run.py
=================
AI Red Teaming Framework -- CLI Entry Point

Author  : Abhinit Sen
GitHub  : github.com/abh1nitsen
LinkedIn: linkedin.com/in/abhinit-sen-63443015
Project : Project 3 of 4 -- AI Governance Portfolio (ISB AMPBA)

WHAT THIS SCRIPT DOES:
    Runs a complete red team assessment against a Groq-hosted language model.
    The assessment tests six governance failure modes relevant to AI systems
    deployed in regulated Indian industry contexts.

    At the end of the run, two reports are saved to reports/:
    - JSON report : machine-readable, compatible with Project 4's dashboard
    - HTML report : human-readable, previewable in any browser

USAGE:
    python setup_and_run.py                    # Run with config.yaml defaults
    python setup_and_run.py --probes 3         # 3 probes per category (faster)
    python setup_and_run.py --category jailbreak   # Run one category only
    python setup_and_run.py --no-html          # JSON report only

PREREQUISITES:
    1. Python 3.9+
    2. pip install -r requirements.txt
    3. GROQ_API_KEY set in environment or .env file

    The framework is completely free to run. The only cost is Groq API tokens,
    which are free on the developer tier for the model volumes used here.
"""

import argparse
import os
import sys

# -- Ensure the project root is on sys.path ----------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# -- Try to load .env if python-dotenv is installed --------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not required; .env is optional


def _load_config(config_path="config/config.yaml"):
    """Load config.yaml and return as a dict."""
    import yaml
    full_path = os.path.join(ROOT, config_path)
    if not os.path.exists(full_path):
        print(f"[FAIL] Config not found: {full_path}")
        sys.exit(1)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(config, args):
    """Apply command-line argument overrides to config dict."""
    if args.probes:
        for cat in config.get("attack_categories", {}).values():
            cat["probes_per_category"] = args.probes
        print(f"[OK] Override: {args.probes} probes per category")

    if args.category:
        valid_cats = set(config.get("attack_categories", {}).keys())
        if args.category not in valid_cats:
            print(f"[FAIL] Unknown category: {args.category}")
            print(f"       Valid options: {', '.join(sorted(valid_cats))}")
            sys.exit(1)
        for cat_name, cat_cfg in config.get("attack_categories", {}).items():
            cat_cfg["enabled"] = (cat_name == args.category)
        print(f"[OK] Override: running category '{args.category}' only")

    if args.no_html:
        config.setdefault("reporting", {})["generate_html"] = False
        print("[OK] Override: HTML report disabled")

    return config


def _check_api_key():
    """Verify GROQ_API_KEY is set. Print a helpful message if not."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print()
        print("[FAIL] GROQ_API_KEY not found in environment.")
        print()
        print("  Option A (local): Create a .env file with:")
        print("    GROQ_API_KEY=your_key_here")
        print()
        print("  Option B (Colab): Add GROQ_API_KEY to Colab Secrets")
        print("    (key icon in the left sidebar)")
        print()
        print("  Get a free Groq API key at: https://console.groq.com/keys")
        sys.exit(1)
    return key


def main():
    parser = argparse.ArgumentParser(
        description="AI Red Teaming Framework -- Red team a Groq-hosted LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--probes", type=int, metavar="N",
        help="Number of probes per category (overrides config.yaml default of 5)"
    )
    parser.add_argument(
        "--category", type=str, metavar="NAME",
        help="Run only one category: regulatory_boundary, pii_extraction, "
             "jailbreak, bias_elicitation, hallucination_induction, or rag_attack"
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML report generation (saves time if you only want JSON)"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)"
    )
    args = parser.parse_args()

    # -- Startup checks -------------------------------------------------------
    _check_api_key()
    config = _load_config(args.config)
    config = _apply_cli_overrides(config, args)

    # -- Run assessment -------------------------------------------------------
    from src.runner.red_team_runner import RedTeamRunner
    from src.reporting.report_generator import ReportGenerator

    runner = RedTeamRunner(config=config, project_root=ROOT)
    result = runner.run(verbose=True)

    # -- Save reports ---------------------------------------------------------
    gen = ReportGenerator(config=config)
    paths = gen.generate(result)

    print("\nReports saved:")
    if "json" in paths:
        print(f"  JSON : {paths['json']}")
    if "html" in paths:
        print(f"  HTML : {paths['html']}")
        print(f"\n  Open the HTML file in a browser to view the full report.")
    print()
    print(f"  AI Red Teaming Framework -- Abhinit Sen -- github.com/abh1nitsen")
    print()

    # -- Exit code reflects verdict ------------------------------------------
    # 0 = ROBUST, 1 = NEEDS REVIEW, 2 = HIGH RISK
    exit_codes = {"ROBUST": 0, "NEEDS REVIEW": 1, "HIGH RISK": 2}
    sys.exit(exit_codes.get(result.verdict, 1))


if __name__ == "__main__":
    main()
