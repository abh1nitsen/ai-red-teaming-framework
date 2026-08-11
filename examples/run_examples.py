"""
examples/run_examples.py
==========================
Demo script showing the three most common ways to use the framework.

Requires GROQ_API_KEY in environment.
Run from the project root: python examples/run_examples.py

Author  : Abhinit Sen
GitHub  : github.com/abh1nitsen
"""

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

def section(title):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Style.RESET_ALL}")

# ---- Example 1: Quick single-category jailbreak test -----------------------

def example_1_jailbreak_only():
    section("Example 1: Quick jailbreak resistance test (2 probes)")
    import yaml
    from src.runner.red_team_runner import RedTeamRunner
    from src.reporting.report_generator import ReportGenerator

    with open(os.path.join(ROOT, "config/config.yaml")) as f:
        config = yaml.safe_load(f)

    # Run jailbreak category only, 2 probes
    for k, v in config["attack_categories"].items():
        v["enabled"] = (k == "jailbreak")
        v["probes_per_category"] = 2

    runner = RedTeamRunner(config=config, project_root=ROOT)
    result = runner.run(verbose=True)
    ReportGenerator(config=config).generate(result)


# ---- Example 2: Full assessment, 3 probes per category --------------------

def example_2_full_assessment_quick():
    section("Example 2: Full red team assessment -- 3 probes per category")
    import yaml
    from src.runner.red_team_runner import RedTeamRunner
    from src.reporting.report_generator import ReportGenerator

    with open(os.path.join(ROOT, "config/config.yaml")) as f:
        config = yaml.safe_load(f)

    for v in config["attack_categories"].values():
        v["probes_per_category"] = 3

    runner = RedTeamRunner(config=config, project_root=ROOT)
    result = runner.run(verbose=True)
    paths = ReportGenerator(config=config).generate(result)

    print(f"\n{Fore.GREEN}HTML report: {paths.get('html', 'not generated')}{Style.RESET_ALL}")


# ---- Example 3: Regulatory boundary focus (most relevant for Indian context)

def example_3_regulatory_focus():
    section("Example 3: Regulatory boundary deep-dive (5 probes)")
    import yaml
    from src.runner.red_team_runner import RedTeamRunner
    from src.reporting.report_generator import ReportGenerator

    with open(os.path.join(ROOT, "config/config.yaml")) as f:
        config = yaml.safe_load(f)

    for k, v in config["attack_categories"].items():
        v["enabled"] = (k == "regulatory_boundary")
        v["probes_per_category"] = 5

    runner = RedTeamRunner(config=config, project_root=ROOT)
    result = runner.run(verbose=True)
    ReportGenerator(config=config).generate(result)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--example", type=int, choices=[1, 2, 3], default=2,
                   help="Which example to run (1=jailbreak, 2=full quick, 3=regulatory)")
    args = p.parse_args()

    {1: example_1_jailbreak_only,
     2: example_2_full_assessment_quick,
     3: example_3_regulatory_focus}[args.example]()
