from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _format_timestamp(value: str | None, note: str | None = None) -> str:
    if value is None:
        return note or "—"
    text = _parse_timestamp(value).strftime("%Y-%m-%d %H:%M:%S %z")
    return text[:-2] + ":" + text[-2:]


def _duration_seconds(session: dict[str, Any]) -> int:
    if session.get("duration_seconds") is not None:
        return int(session["duration_seconds"])
    if session.get("start") is None or session.get("stop") is None:
        raise ValueError("Stopped sessions require timestamps or an explicit retrospective duration")
    seconds = (_parse_timestamp(session["stop"]) - _parse_timestamp(session["start"])).total_seconds()
    if seconds < 0 or not seconds.is_integer():
        raise ValueError("Session duration must be a non-negative whole number of seconds")
    return int(seconds)


def _clock(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _session_rows(sessions: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for session in sessions:
        note = session.get("timestamp_note")
        rows.append(
            "| "
            + " | ".join(
                [
                    _cell(_format_timestamp(session.get("start"), note)),
                    _cell(_format_timestamp(session.get("stop"), note)),
                    _clock(_duration_seconds(session)),
                    _cell(session["activity"]),
                    _cell(session["evidence"]),
                    _cell(session["human_verification"]),
                ]
            )
            + " |"
        )
    return rows


def _agent_rows(sessions: list[dict[str, Any]]) -> list[str]:
    return [
        "| "
        + " | ".join(
            [
                _format_timestamp(session["start"]),
                _format_timestamp(session["stop"]),
                _clock(_duration_seconds(session)),
                _cell(session["agent"]),
                _cell(session["activity"]),
                _cell(session["output"]),
            ]
        )
        + " |"
        for session in sessions
    ]


def render(ledger: dict[str, Any]) -> str:
    if ledger.get("schema_version") != 1:
        raise ValueError("Unsupported time-ledger schema")
    counted = sum(_duration_seconds(session) for session in ledger["counted_sessions"])
    executive = sum(_duration_seconds(session) for session in ledger["executive_sessions"])
    preparation = sum(_duration_seconds(session) for session in ledger["preparation_sessions"])
    budgets = ledger["budgets_seconds"]
    timer = ledger["timer"]
    state = timer["state"]
    if state not in {"RUNNING", "STOPPED"}:
        raise ValueError("Timer state must be RUNNING or STOPPED")
    active = (
        f"{timer['activity']}, started `{_format_timestamp(timer['start'])}`"
        if state == "RUNNING"
        else "none"
    )
    lines = [
        "<!-- GENERATED from time-ledger.json by frame_stage_binding.time_ledger; do not edit. -->",
        "# MATS application time log",
        "",
        "Raw timestamps in `time-ledger.json` are authoritative. This Markdown view and all",
        "durations/totals are generated programmatically. Times use Asia/Kolkata unless noted.",
        "",
        "## Timer state",
        "",
        f"- State: **{state}**",
        f"- Active session: {active}",
        f"- Current official project: {ledger['official_project']}",
        "",
        "## Totals",
        "",
        "| Budget | Used | Target/cap | Remaining |",
        "|---|---:|---:|---:|",
        f"| Counted project work | {_clock(counted)} | target {_clock(budgets['counted_target'])}; hard cap {_clock(budgets['counted_hard_cap'])} | {_clock(max(0, budgets['counted_target'] - counted))} to target; {_clock(max(0, budgets['counted_hard_cap'] - counted))} to cap |",
        f"| Executive-summary extension | {_clock(executive)} | cap {_clock(budgets['executive_cap'])} | {_clock(max(0, budgets['executive_cap'] - executive))} |",
        f"| Uncounted preparation (informational) | {_clock(preparation)} | no cap | — |",
        "",
        "Open running time is added only when the session is stopped from a raw timestamp.",
        "",
        "## Counted project sessions",
        "",
        "| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |",
        "|---|---|---:|---|---|---|",
        *_session_rows(ledger["counted_sessions"]),
    ]
    if state == "RUNNING" and timer["bucket"] == "counted":
        lines.append(
            f"| {_format_timestamp(timer['start'])} | **OPEN** | **RUNNING** | {_cell(timer['activity'])} | {_cell(timer['evidence'])} | {_cell(timer['human_verification'])} |"
        )
    lines.extend(
        [
            "",
            "## Executive-summary extension sessions",
            "",
            "| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |",
            "|---|---|---:|---|---|---|",
            *_session_rows(ledger["executive_sessions"]),
        ]
    )
    if state == "RUNNING" and timer["bucket"] == "executive":
        lines.append(
            f"| {_format_timestamp(timer['start'])} | **OPEN** | **RUNNING** | {_cell(timer['activity'])} | {_cell(timer['evidence'])} | {_cell(timer['human_verification'])} |"
        )
    lines.extend(
        [
            "",
            "## Uncounted preparation sessions",
            "",
            "| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |",
            "|---|---|---:|---|---|---|",
            *_session_rows(ledger["preparation_sessions"]),
            "",
            "## Abandoned/reset project sessions",
            "",
            "| Start timestamp | Stop timestamp | Duration | Activity | Evidence/output | Human verification |",
            "|---|---|---:|---|---|---|",
            *_session_rows(ledger["abandoned_sessions"]),
            "",
            "## Counting convention for agent-assisted work",
            "",
            "Count all applicant-active work: decisions, design, project-specific reading,",
            "prompting/guiding/debugging agents, code inspection, experiment supervision,",
            "analysis, verification, and own-voice writing. Generic environment setup is",
            "uncounted. Detached agent execution and passive runtime are uncounted only while",
            "the applicant is genuinely inactive or doing unrelated work, and are logged below.",
            "",
            "## Agent execution sessions (informational, uncounted)",
            "",
            "| Start timestamp | Stop timestamp | Duration | Agent | Activity | Output/artifact |",
            "|---|---|---:|---|---|---|",
            *_agent_rows(ledger["agent_execution_sessions"]),
            "",
            "## Logging protocol",
            "",
            "- Use `frame-stage-time start` and `frame-stage-time stop`; never calculate or type durations/totals manually.",
            "- Start only on the applicant's explicit instruction and stop on an explicit stop instruction.",
            "- Do not overlap counted sessions. Stop for breaks.",
            "- Model execution additionally requires the phase-specific active-session label enforced by the runner.",
            "- The retrospective estimate is the sole timestamp-free exception and is explicitly marked in the raw ledger.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary path already exists: {temporary}")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save(ledger_path: Path, output_path: Path, ledger: dict[str, Any]) -> None:
    _write_atomic(ledger_path, json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    _write_atomic(output_path, render(ledger))


def _now(ledger: dict[str, Any]) -> str:
    return datetime.now(ZoneInfo(ledger["timezone"])).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maintain and render the MATS time ledger")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("docs/research-log/admin/time-ledger.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research-log/admin/time-log.md"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("render")
    start = subparsers.add_parser("start")
    start.add_argument("--bucket", choices=["counted", "executive"], required=True)
    start.add_argument("--activity", required=True)
    start.add_argument("--evidence", default="In progress")
    start.add_argument("--human-verification", required=True)
    stop = subparsers.add_parser("stop")
    stop.add_argument("--evidence", required=True)
    stop.add_argument("--human-verification", required=True)
    reconcile = subparsers.add_parser("reconcile-away")
    reconcile.add_argument("--session-index", type=int, required=True)
    reconciliation = reconcile.add_mutually_exclusive_group(required=True)
    reconciliation.add_argument("--left-immediately", action="store_true")
    reconciliation.add_argument("--active-minutes", type=int)
    reconcile.add_argument("--reason", required=True)
    reconcile.add_argument(
        "--detached-agent", default="Codex and read-only reviewers"
    )
    reconcile.add_argument(
        "--detached-activity",
        default="Detached implementation, testing, and audits while the applicant was away",
    )
    reconcile.add_argument("--detached-evidence")
    agent = subparsers.add_parser("record-agent-session")
    agent.add_argument("--start-from-last-counted-stop", action="store_true")
    agent.add_argument("--agent", required=True)
    agent.add_argument("--activity", required=True)
    agent.add_argument("--evidence", required=True)
    retrospective = subparsers.add_parser("record-retrospective")
    retrospective.add_argument("--minutes", type=int, required=True)
    retrospective.add_argument("--activity", required=True)
    retrospective.add_argument("--evidence", required=True)
    retrospective.add_argument("--human-verification", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = json.loads(args.ledger.read_text())
    if args.command == "render":
        _write_atomic(args.output, render(ledger))
        return
    if args.command == "reconcile-away":
        if ledger["timer"]["state"] != "STOPPED":
            raise ValueError("Stop the active timer before reconciling away time")
        session = ledger["counted_sessions"][args.session_index]
        wall_duration = int(
            (_parse_timestamp(session["stop"]) - _parse_timestamp(session["start"])).total_seconds()
        )
        if args.left_immediately:
            active_seconds = 0
            rule = "left_immediately"
            estimate_text = "leaving immediately"
        else:
            if args.active_minutes is None or args.active_minutes < 0:
                raise ValueError("--active-minutes must be non-negative")
            active_seconds = args.active_minutes * 60
            rule = "active_minutes_then_away"
            estimate_text = f"approximately {args.active_minutes} active minutes"
        if active_seconds > wall_duration:
            raise ValueError("Applicant-active estimate exceeds the wall interval")
        previous = session.get("reconciliation")
        if previous is not None and previous != {"rule": rule, "reason": args.reason}:
            session.setdefault("reconciliation_history", []).append(previous)
        session["duration_seconds"] = active_seconds
        session["wall_duration_seconds"] = wall_duration
        session["reconciliation"] = {
            "rule": rule,
            "reason": args.reason,
        }
        base_evidence = session["evidence"].partition("; gross interval")[0]
        base_evidence = base_evidence.partition("; applicant-active duration")[0]
        session["evidence"] = (
            base_evidence + "; applicant-active duration reconciled from applicant report"
        )
        session["human_verification"] = (
            "Applicant reported being away after starting the timer and estimated "
            f"{estimate_text}; the corrected estimate is counted and the remainder is excluded"
        )
        away_start = (_parse_timestamp(session["start"]) + timedelta(seconds=active_seconds)).isoformat()
        detached = next(
            (
                agent_session
                for agent_session in ledger["agent_execution_sessions"]
                if agent_session.get("source_counted_session_index") == args.session_index
            ),
            None,
        )
        detached_values = {
            "start": away_start,
            "stop": session["stop"],
            "agent": args.detached_agent,
            "activity": args.detached_activity,
            "output": args.detached_evidence or session["evidence"],
            "source_counted_session_index": args.session_index,
        }
        if detached is None:
            ledger["agent_execution_sessions"].append(detached_values)
        else:
            detached.update(detached_values)
        save(args.ledger, args.output, ledger)
        return
    if args.command == "record-agent-session":
        if ledger["timer"]["state"] != "STOPPED":
            raise ValueError("Stop the applicant timer before recording detached work")
        if not args.start_from_last_counted_stop:
            raise ValueError("Detached sessions must derive their start from a raw ledger stop")
        stopped = [
            session for session in ledger["counted_sessions"] if session.get("stop")
        ]
        if not stopped:
            raise ValueError("No stopped counted session is available")
        start = max(stopped, key=lambda session: _parse_timestamp(session["stop"]))["stop"]
        stop = _now(ledger)
        ledger["agent_execution_sessions"].append(
            {
                "start": start,
                "stop": stop,
                "agent": args.agent,
                "activity": args.activity,
                "output": args.evidence,
            }
        )
        save(args.ledger, args.output, ledger)
        return
    if args.command == "record-retrospective":
        if ledger["timer"]["state"] != "STOPPED":
            raise ValueError("Stop the active timer before recording retrospective time")
        if args.minutes <= 0:
            raise ValueError("--minutes must be positive")
        ledger["counted_sessions"].append(
            {
                "start": None,
                "stop": None,
                "duration_seconds": args.minutes * 60,
                "activity": args.activity,
                "evidence": args.evidence,
                "human_verification": args.human_verification,
                "timestamp_note": "Retrospective applicant estimate; exact timestamps unavailable",
            }
        )
        save(args.ledger, args.output, ledger)
        return
    if args.command == "start":
        if ledger["timer"]["state"] != "STOPPED":
            raise ValueError("A timer session is already running")
        ledger["timer"] = {
            "state": "RUNNING",
            "bucket": args.bucket,
            "start": _now(ledger),
            "activity": args.activity,
            "evidence": args.evidence,
            "human_verification": args.human_verification,
        }
        save(args.ledger, args.output, ledger)
        return
    if ledger["timer"]["state"] != "RUNNING":
        raise ValueError("No timer session is running")
    timer = ledger["timer"]
    session = {
        "start": timer["start"],
        "stop": _now(ledger),
        "activity": timer["activity"],
        "evidence": args.evidence,
        "human_verification": args.human_verification,
    }
    target = "counted_sessions" if timer["bucket"] == "counted" else "executive_sessions"
    ledger[target].append(session)
    ledger["timer"] = {"state": "STOPPED", "bucket": None, "start": None, "activity": None}
    save(args.ledger, args.output, ledger)


if __name__ == "__main__":
    main()
