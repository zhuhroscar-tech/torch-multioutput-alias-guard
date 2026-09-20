"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is
installed."""
from __future__ import annotations

import json

import pytest

import torch_multioutput_alias_guard.core as core
from torch_multioutput_alias_guard.cli import main


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-multioutput-alias-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert code in (0, 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_torch_unavailable_json_output_reports_error_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload == {"error": "torch is required for diagnosis and guarding"}
    assert code == 2


def test_torch_unavailable_text_output_shows_fail_headline_and_exit_2(capsys, monkeypatch):
    def _raise(*args, **kwargs):
        raise core.TorchUnavailableError("torch is required for diagnosis and guarding")

    monkeypatch.setattr(core, "diagnose", _raise)
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] torch unavailable: torch is required for diagnosis and guarding" in out
    assert code == 2


def _fake_report(aminmax_bug, slogdet_bug, guard_ok, aminmax_guard_raised=True, slogdet_guard_raised=True):
    return {
        "torch_version": "0.0.0-fake",
        "issue_urls": ["https://github.com/pytorch/pytorch/issues/195338"],
        "aminmax_cases": [
            {
                "values": [3.0, 1.0, 4.0],
                "native_aliased_result": 4.0 if aminmax_bug else 1.0,
                "native_aliased_raised": False,
                "reference_min": 1.0,
                "reference_max": 4.0,
                "native_result_is_wrong": aminmax_bug,
                "guard_raised": aminmax_guard_raised,
            }
        ],
        "slogdet_cases": [
            {
                "matrix": [[2.0, 0.0], [0.0, 2.0]],
                "native_aliased_result": 1.386 if slogdet_bug else 1.0,
                "native_aliased_raised": False,
                "reference_sign": 1.0,
                "reference_logabsdet": 1.386,
                "native_result_lost_sign": slogdet_bug,
                "guard_raised": slogdet_guard_raised,
            }
        ],
        "any_aminmax_alias_bug": aminmax_bug,
        "any_slogdet_alias_bug": slogdet_bug,
        "guard_fully_correct": guard_ok,
    }


def test_no_bugs_shows_info_messages_not_fail(capsys, monkeypatch):
    monkeypatch.setattr(
        core, "diagnose", lambda **kwargs: _fake_report(False, False, True)
    )
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[i] no torch.aminmax aliasing divergence reproduced" in out
    assert "[i] no torch.linalg.slogdet aliasing divergence reproduced" in out
    assert "[X]" not in out.split("\n")[1]  # no fail headline for the bug-reproduction lines
    assert code == 0


def test_guard_did_not_raise_label_shown_when_guard_ineffective(capsys, monkeypatch):
    """The per-case table's "GUARD-DID-NOT-RAISE" label was never
    actually rendered by any real run on this host -- every real
    repro has safe_aminmax()/safe_slogdet() succeed (raise) for every
    probed case, so this branch is only reachable by construction, not
    observation."""
    monkeypatch.setattr(
        core, "diagnose", lambda **kwargs: _fake_report(True, True, False, aminmax_guard_raised=False)
    )
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "GUARD-DID-NOT-RAISE" in out
    assert "[X] guard did NOT raise on at least one aliased case" in out
    assert code == 1


def test_bugs_reproduced_shows_fail_headlines(capsys, monkeypatch):
    monkeypatch.setattr(
        core, "diagnose", lambda **kwargs: _fake_report(True, True, True)
    )
    code = main(["--no-color"])
    out = capsys.readouterr().out
    assert "[X] torch.aminmax(out=(t, t)) same-tensor aliasing bug reproduced on this host" in out
    assert "[X] torch.linalg.slogdet(out=(t, t)) same-tensor aliasing bug reproduced on this host" in out
    assert "[OK] safe_aminmax()/safe_slogdet() raise before computing on every aliased case" in out
    assert code == 0
