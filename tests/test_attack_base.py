"""
tests/test_attack_base.py
==========================
Tests for AttackBase shared logic: probe loading, selection, aggregation, risk rating.
All tests are offline -- no API calls.
"""
import json, os, tempfile, pytest
from unittest.mock import MagicMock, patch
from src.attacks.attack_base import AttackBase, ProbeResult, CategoryResult
from src.utils.groq_client import GroqClient


class ConcreteAttack(AttackBase):
    """Minimal concrete subclass for testing the base class."""
    CATEGORY_NAME = "test_category"

    def __init__(self, library_file="", **kwargs):
        self.LIBRARY_FILE = library_file
        super().__init__(**kwargs)

    def _score_response(self, probe, response):
        return 0.8, "test reason"


def _make_library(probes, tmpdir):
    path = os.path.join(tmpdir, "probes.json")
    with open(path, "w") as f:
        json.dump({"probes": probes}, f)
    return path


class TestProbeLoading:
    def test_loads_probes_from_valid_json_library(self, tmp_path):
        probes = [{"id": "p1", "name": "Probe 1", "prompt": "Hello?"}]
        lib = _make_library(probes, tmp_path)
        attack = ConcreteAttack(library_file=lib, project_root="",
                                config={}, target_client=MagicMock(),
                                judge_client=MagicMock())
        attack.LIBRARY_FILE = lib
        loaded = attack._load_probes()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "p1"

    def test_returns_empty_list_when_library_file_not_set(self, tmp_path):
        attack = ConcreteAttack(library_file="", project_root=str(tmp_path),
                                config={}, target_client=MagicMock(),
                                judge_client=MagicMock())
        assert attack._load_probes() == []

    def test_returns_empty_list_when_library_file_missing(self, tmp_path):
        attack = ConcreteAttack(library_file="nonexistent.json",
                                project_root=str(tmp_path),
                                config={}, target_client=MagicMock(),
                                judge_client=MagicMock())
        assert attack._load_probes() == []

    def test_caches_probes_after_first_load(self, tmp_path):
        probes = [{"id": "p1", "prompt": "Q?"}]
        lib = _make_library(probes, tmp_path)
        attack = ConcreteAttack(library_file=lib, project_root="",
                                config={}, target_client=MagicMock(),
                                judge_client=MagicMock())
        attack.LIBRARY_FILE = lib
        attack._load_probes()
        attack._load_probes()   # Second call should use cache
        assert attack._probes is not None


class TestProbeSelection:
    def test_selects_configured_number_of_probes(self, tmp_path):
        probes = [{"id": f"p{i}", "prompt": f"Q{i}?"} for i in range(10)]
        lib = _make_library(probes, tmp_path)
        attack = ConcreteAttack(library_file=lib, project_root="",
                                config={"attack_categories": {
                                    "test_category": {"probes_per_category": 3,
                                                      "shuffle": False}
                                }},
                                target_client=MagicMock(),
                                judge_client=MagicMock())
        attack.LIBRARY_FILE = lib
        selected = attack._select_probes()
        assert len(selected) == 3

    def test_selects_all_when_fewer_probes_than_configured(self, tmp_path):
        probes = [{"id": "p1", "prompt": "Q?"}]
        lib = _make_library(probes, tmp_path)
        attack = ConcreteAttack(library_file=lib, project_root="",
                                config={"attack_categories": {
                                    "test_category": {"probes_per_category": 5,
                                                      "shuffle": False}
                                }},
                                target_client=MagicMock(),
                                judge_client=MagicMock())
        attack.LIBRARY_FILE = lib
        selected = attack._select_probes()
        assert len(selected) == 1


class TestAggregation:
    def _make_attack(self):
        return ConcreteAttack(
            config={"risk_thresholds": {"high_above": 0.50, "medium_above": 0.20}},
            target_client=MagicMock(),
            judge_client=MagicMock(),
        )

    def test_attack_success_rate_computed_correctly(self):
        attack = self._make_attack()
        results = [
            ProbeResult(attack_succeeded=True),
            ProbeResult(attack_succeeded=True),
            ProbeResult(attack_succeeded=False),
            ProbeResult(attack_succeeded=False),
        ]
        cat = attack._aggregate(results)
        assert cat.attack_success_rate == 0.5
        assert cat.successful_attacks == 2
        assert cat.probes_run == 4

    def test_risk_rated_high_when_over_fifty_percent_attacks_succeed(self):
        attack = self._make_attack()
        results = [ProbeResult(attack_succeeded=True)] * 4 + \
                  [ProbeResult(attack_succeeded=False)]
        cat = attack._aggregate(results)
        assert cat.risk_rating == "HIGH"

    def test_risk_rated_medium_when_between_twenty_and_fifty_percent(self):
        attack = self._make_attack()
        results = [ProbeResult(attack_succeeded=True)] * 2 + \
                  [ProbeResult(attack_succeeded=False)] * 5
        cat = attack._aggregate(results)
        assert cat.risk_rating == "MEDIUM"

    def test_risk_rated_low_when_under_twenty_percent_attacks_succeed(self):
        attack = self._make_attack()
        results = [ProbeResult(attack_succeeded=True)] + \
                  [ProbeResult(attack_succeeded=False)] * 9
        cat = attack._aggregate(results)
        assert cat.risk_rating == "LOW"

    def test_empty_results_return_zero_success_rate(self):
        attack = self._make_attack()
        cat = attack._aggregate([])
        assert cat.attack_success_rate == 0.0
        assert cat.probes_run == 0


class TestDisabledCategory:
    def test_disabled_category_returns_immediately_without_running_probes(self):
        attack = ConcreteAttack(
            config={"attack_categories": {
                "test_category": {"enabled": False}
            }},
            target_client=MagicMock(),
            judge_client=MagicMock(),
        )
        result = attack.run(verbose=False)
        assert result.error is not None
        assert result.probes_run == 0


class TestProbeResult:
    def test_to_dict_contains_required_keys(self):
        pr = ProbeResult(
            probe_id="p1", probe_name="Test probe",
            probe_text="Prompt?", response="Answer.",
            defence_score=0.8, attack_succeeded=False, reason="OK"
        )
        d = pr.to_dict()
        for k in ("probe_id", "probe_name", "defence_score",
                  "attack_succeeded", "reason", "error"):
            assert k in d

    def test_response_truncated_in_dict_output(self):
        pr = ProbeResult(response="x" * 1000)
        d = pr.to_dict()
        assert len(d["response_preview"]) <= 400
