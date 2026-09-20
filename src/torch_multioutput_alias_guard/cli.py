"""Command-line interface: run the from-scratch diagnosis of the
torch.aminmax / torch.linalg.slogdet same-tensor output-aliasing
correctness bug class against the currently installed torch build,
using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-multioutput-alias-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "torch.aminmax and torch.linalg.slogdet silently compute a "
            "wrong result when the SAME tensor is passed for two "
            "different out= slots (pytorch/pytorch#195338 and this "
            "candidate's own LAPACK-backed slogdet finding) -- and "
            "verify the safe_aminmax()/safe_slogdet() guards raise "
            "BEFORE any wrong value is computed. Never trusts a cached "
            "or previously-reported result, always re-runs the repro "
            "on THIS host's actual installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-multioutput-alias-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_aminmax_alias_bug"]:
        print(status_headline(style, "fail", "torch.aminmax(out=(t, t)) same-tensor aliasing bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no torch.aminmax aliasing divergence reproduced on this host's installed torch build"))

    if report["any_slogdet_alias_bug"]:
        print(status_headline(style, "fail", "torch.linalg.slogdet(out=(t, t)) same-tensor aliasing bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no torch.linalg.slogdet aliasing divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_aminmax()/safe_slogdet() raise before computing on every aliased case"))
    else:
        print(status_headline(style, "fail", "guard did NOT raise on at least one aliased case"))

    section("aminmax cases (values -> native aliased result vs reference min/max, guard behavior)")
    for c in report["aminmax_cases"]:
        native_flag = "WRONG" if c["native_result_is_wrong"] else ("raised" if c["native_aliased_raised"] else "ok")
        guard_flag = "guard-raised" if c["guard_raised"] else "GUARD-DID-NOT-RAISE"
        print_fields(
            [
                (
                    f"values={c['values']}",
                    f"native_aliased={c['native_aliased_result']!r}  "
                    f"ref_min={c['reference_min']}  ref_max={c['reference_max']}  "
                    f"{native_flag:6s}  {guard_flag}",
                )
            ]
        )

    section("slogdet cases (matrix -> native aliased result vs reference sign/logabsdet, guard behavior)")
    for c in report["slogdet_cases"]:
        native_flag = "LOST-SIGN" if c["native_result_lost_sign"] else ("raised" if c["native_aliased_raised"] else "ok")
        guard_flag = "guard-raised" if c["guard_raised"] else "GUARD-DID-NOT-RAISE"
        print_fields(
            [
                (
                    f"matrix={c['matrix']}",
                    f"native_aliased={c['native_aliased_result']!r}  "
                    f"ref_sign={c['reference_sign']}  ref_logabsdet={c['reference_logabsdet']:.4f}  "
                    f"{native_flag:10s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
