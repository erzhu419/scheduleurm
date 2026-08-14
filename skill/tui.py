"""Interactive TUI for `scheduler status` — sortable table, filter, auto-refresh.
Run via `python ~/.claude/skills/scheduler/scheduler.py tui`.

Node telemetry is read from the watcher's atomic probe cache first.  If that
cache is unavailable, at most one live probe runs in the background; slow SSH
timeouts never blank or block the UI.  Sort/filter operate on the cached
snapshot — instant response.

Keys:
  r / q / a    → filter to running / queued / all active
  f            → focus filter input (substring match against id/project/location/owner/sig/desc)
  1..9         → sort by column (id / status / node / project / owner / runtime / vram / ram / eta)
  R            → reverse sort direction
  p / P        → bump task priority up / down (only for queued tasks)
  l            → show current row's task-log tail
  c            → copy current row's task id to clipboard (paste e.g. into `cancel`/`show`)
  ctrl+r       → force refresh now
  ctrl+c       → quit
You can also CLICK a column header to sort by it (click again = reverse).
Drag a column header to reorder columns; drag the visible │ separator to resize it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    from textual import work
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container
    from textual.reactive import reactive
    from textual.screen import ModalScreen
    from textual.widgets import DataTable, Footer, Header, Input, Static
    from textual.widgets import TextArea
    from textual.worker import Worker, WorkerState
except ImportError:
    sys.exit("textual not installed. Run: pip install --user textual")

sys.path.insert(0, str(Path(__file__).parent))
import scheduler as sch  # noqa: E402

_SCHED_SOURCE = Path(getattr(sch, "__file__", Path(__file__).with_name("scheduler.py"))).resolve()
_TUI_SOURCE = Path(__file__).resolve()
_NODE_INVENTORY_MODULE = sys.modules.get("scheduler_node.inventory")
_NODE_INVENTORY_SOURCE = Path(
    getattr(_NODE_INVENTORY_MODULE, "__file__", _SCHED_SOURCE)
).resolve()
_SOURCE_MTIMES = {}
PROBE_UNKNOWN_WARN_S = max(0, int(os.environ.get("SCHEDULEURM_TUI_PROBE_UNKNOWN_WARN_S", "180")))
TERMINAL_TASK_STATUSES = frozenset({"done", "failed", "cancelled"})
for _src in (_SCHED_SOURCE, _TUI_SOURCE, _NODE_INVENTORY_SOURCE):
    try:
        _SOURCE_MTIMES[_src] = _src.stat().st_mtime_ns
    except OSError:
        _SOURCE_MTIMES[_src] = 0


def _scheduler_source_changed() -> bool:
    for src, old_mtime in _SOURCE_MTIMES.items():
        try:
            if src.stat().st_mtime_ns != old_mtime:
                return True
        except OSError:
            return True
    return False


def _fmt_min(secs):
    if secs is None or secs < 0: return "-"
    if secs < 60: return f"{int(secs)}s"
    if secs < 3600: return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


def _probe_unknown_age(task, now=None):
    if not task or not task.get("probe_unknown_since"):
        return None
    now = now or time.time()
    try:
        return max(0, now - float(task.get("probe_unknown_since") or 0))
    except Exception:
        return None


def _display_status(task, now=None):
    status = task.get("status") or "-"
    age = _probe_unknown_age(task, now)
    if status == "running" and age is not None and age >= PROBE_UNKNOWN_WARN_S:
        return "running?"
    return status


def _display_node(task, now=None):
    node = sch._format_task_location(task)
    age = _probe_unknown_age(task, now)
    if age is not None and age >= PROBE_UNKNOWN_WARN_S:
        return f"{node} probe?{_fmt_min(age)}"
    return node


def _int_or_default(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_eta(t, hist):
    if t.get("status") in TERMINAL_TASK_STATUSES:
        return "-"
    eta = _int_or_default(t.get("eta_seconds"), 0)
    source = sch._eta_source_base(t.get("eta_source"))
    tag = sch._eta_source_tag(t.get("eta_source")) if hasattr(sch, "_eta_source_tag") else "est"
    current = _int_or_default(t.get("runtime_current_unit"), 0)
    total = _int_or_default(t.get("runtime_total_units"), 0)
    progress = f" {current}/{total}" if current > 0 and total >= current else ""
    if eta > 0:
        if source in ("runtime_history_overrun", "duration_ewma_overrun"):
            return "? hist"
        return f"~{_fmt_min(eta)} {tag}{progress}"
    if t.get("status") == "running" and total > 0 and current >= total:
        return "finishing live"
    sig = t.get("signature") or ""
    h = hist.get(sig, {})
    if isinstance(h, int): h = {"vram_mb": h}
    expected = h.get("dur_s_ewma", 0)
    if t.get("status") == "running":
        if not t.get("started_at"): return "-"
        elapsed = time.time() - t["started_at"]
        # Auto-adopted: started_at = adopt time (not real launch), and historical EWMA was
        # itself measured across adopt cycles — both numbers are structurally lower than
        # truth. Pretending to predict ETA here would be misleading; just show elapsed-since-adopt
        # with a ? marker so the user knows total duration is unknown.
        if t.get("auto_adopted"):
            return f"{_fmt_min(elapsed)}+ ?"
        if not expected:
            return f"{_fmt_min(elapsed)}+"
        if elapsed >= expected:
            # Ran past the EWMA prediction. Showing "~0s (100%)" was misleading — the run is
            # not 100% done, the prediction was wrong. Surface the overrun instead so user can
            # tell at a glance "this is way past what history said".
            over = elapsed - expected
            return f"+{_fmt_min(over)} over"
        remaining = expected - elapsed
        pct = int(elapsed / expected * 100)
        return f"~{_fmt_min(remaining)} ({pct}%)"
    if t.get("status") == "queued":
        return f"~{_fmt_min(expected)}" if expected else "?"
    if t.get("started_at") and t.get("finished_at"):
        return _fmt_min(t["finished_at"] - t["started_at"])
    return "-"


_NODE_SUMMARY_HIDDEN_NAMES = {"zhengliang-hpc"}


def _node_display_name(n):
    name = str(n.get("name") or "")
    return "node007" if name == "node007-direct" else name


def _node_summary_visible(n):
    return str(n.get("name") or "") not in _NODE_SUMMARY_HIDDEN_NAMES


def _node_sort_key(n):
    name = str(n.get("name") or "")
    gpu_order = {
        "local": 0,
        "jtl110gpu": 1,
        "jtl110gpu2": 2,
        "jtl311linux": 3,
        "node007": 4,
    }
    cpu_order = {
        "jtl110cpu": 0,
        "jtl110cpu2": 1,
        "node001": 10,
        "node002": 11,
        "node003": 12,
        "node004": 13,
        "node005": 14,
        "node006": 15,
    }
    if name in gpu_order:
        return (0, gpu_order[name], _node_display_name(n))
    if name in cpu_order:
        return (1, cpu_order[name], _node_display_name(n))
    if n.get("gpus"):
        return (0, 50, _node_display_name(n))
    if name.startswith("node"):
        return (1, 50, _node_display_name(n))
    return (8, 0, _node_display_name(n))


def _node_tail_summary(n):
    reservation_s = ""
    if n.get("measurement_reservation_active"):
        reservation = n.get("measurement_reservation") or {}
        purpose = str(reservation.get("purpose") or "calibration")
        reservation_s = f"MEASUREMENT-RESERVED({purpose})"
    load = n.get("observed_loadavg", n.get("loadavg"))
    load_s = f"load={load:.1f}" if isinstance(load, (int, float)) else ""
    host_cpu = n.get("host_cpu_load_pct")
    if host_cpu is not None:
        wsl_load = n.get("wsl_loadavg")
        if isinstance(wsl_load, (int, float)):
            load_s = f"wsl_load={wsl_load:.1f},host_cpu={int(host_cpu)}%"
        else:
            load_s = f"host_cpu={int(host_cpu)}%"
    if n.get("probe_fallback"):
        load_s = (load_s + "," if load_s else "") + str(n.get("probe_fallback"))
    ram_s = sch._format_node_ram_summary(n)
    total_cpu = n.get("total_cpu", "?")
    observed_free = n.get("observed_free_cpu")
    slot_free = n.get("cpu_slot_free", n.get("free_cpu"))
    guard_free = n.get("cpu_hard_free")
    startup_reserved = n.get("cpu_hard_reserved")
    if observed_free is not None:
        controls = []
        if slot_free is not None and slot_free != observed_free:
            controls.append(f"slot={slot_free}")
        if guard_free is not None and guard_free != observed_free:
            controls.append(f"guard={guard_free}")
        if startup_reserved:
            controls.append(f"startup={startup_reserved}")
        control_s = f"({','.join(controls)})" if controls else ""
        cpu_s = f"cpu={observed_free}/{total_cpu} live{control_s}"
    else:
        cpu_s = f"cpu={n.get('free_cpu','?')}/{total_cpu}"
    claim_s = sch._format_node_claim_summary(n)
    claim_s = claim_s.strip() if claim_s else ""
    return "  ".join(
        s for s in (reservation_s, cpu_s, load_s, ram_s, claim_s) if s
    )


def _node_summary_line(nodes):
    if not nodes: return "(probe pending...)"
    ordered = sorted((n for n in nodes if _node_summary_visible(n)), key=_node_sort_key)
    if not ordered:
        return "(no visible nodes)"
    name_w = max(11, *(len(_node_display_name(n)) for n in ordered))
    lines = [
        f"{'node':<{name_w}} {'gpu':<5} {'used/total':>13} {'free':>9} {'mem':>5} {'util':>9}  resources"
    ]
    for n in ordered:
        name = _node_display_name(n) or "?"
        if not n.get("alive"):
            # Defense in depth: error strings often contain ssh argv like ['ssh', '-o', ...]
            # which Rich parses as markup tags. Strip brackets even though markup is disabled.
            err = (n.get("error", "?") or "?")[:72].replace("[", "(").replace("]", ")")
            state = "WAIT" if n.get("probe_pending") else "DOWN"
            lines.append(
                f"{name:<{name_w}} {state:<5} {'-':>13} {'-':>9} {'-':>5} {'-':>9}  {err}"
            )
            continue
        tail = _node_tail_summary(n)
        gpus = sorted(n.get("gpus") or [], key=lambda g: _int_or_default(g.get("idx"), 999))
        if not gpus:
            lines.append(
                f"{name:<{name_w}} {'cpu':<5} {'-':>13} {'-':>9} {'-':>5} {'-':>9}  {tail}".rstrip()
            )
            continue

        for i, g in enumerate(gpus):
            used_mb = _int_or_default(g.get("used_mb"), 0)
            total_mb = _int_or_default(g.get("total_mb"), 0)
            free_mb = _int_or_default(g.get("free_mb"), 0)
            mem_pct = int(round(used_mb * 100 / max(total_mb, 1)))
            util = f"{_int_or_default(g.get('util_pct'), 0)}%"
            cu = g.get("util_pct_compute")
            if cu is not None:
                util = f"{_int_or_default(g.get('util_pct'), 0)}/{_int_or_default(cu, 0)}%"
            line_name = name if i == 0 else ""
            line_tail = tail if i == 0 else ""
            gpu_label = f"GPU{g.get('idx', '?')}"
            used_total = f"{sch._format_mem_gb(used_mb)}/{sch._format_mem_gb(total_mb)}"
            lines.append(
                f"{line_name:<{name_w}} {gpu_label:<5} "
                f"{used_total:>13} {sch._format_mem_gb(free_mb):>9} "
                f"{str(mem_pct) + '%':>5} {util:>9}  {line_tail}".rstrip()
            )
    return "\n".join(lines)


def _watcher_status_line():
    watcher_state = Path(getattr(sch, "WATCHER_STATE", sch.STATE_DIR / ".watcher_state.json"))
    try:
        raw = json.loads(watcher_state.read_text())
    except FileNotFoundError:
        return "WATCHER DOWN: no watcher state; queue statuses may be stale"
    except Exception as e:
        return f"WATCHER UNKNOWN: cannot read watcher state ({str(e)[:80]})"
    if not isinstance(raw, dict):
        return "WATCHER UNKNOWN: invalid watcher state"
    pid = raw.get("pid")
    try:
        alive = bool(pid) and Path(f"/proc/{int(pid)}").exists()
    except (TypeError, ValueError):
        alive = False
    if not alive:
        return f"WATCHER DOWN: pid={pid or '?'} is not alive; queue statuses may be stale"
    now = time.time()
    progress_ts = float(raw.get("last_progress_ts") or 0)
    last_update = max(
        progress_ts,
        float(raw.get("last_resource_log_ts") or 0),
        float(raw.get("last_heartbeat_ts") or 0),
        float(raw.get("started_at") or 0),
    )
    resource_interval = int(raw.get("resource_log_interval") or 0)
    stale_after = max(600, 2 * resource_interval + 120) if resource_interval else 900
    if last_update and now - last_update > stale_after:
        phase = str(raw.get("phase") or "unknown")
        return (
            f"WATCHER STALE: pid={pid}, phase={phase}, "
            f"last progress {_fmt_min(now - last_update)} ago"
        )
    return ""


COLUMNS = [
    ("id", "id", 6),
    ("status", "status", 9),
    ("node", "location", 24),
    ("project", "project", 12),
    ("owner", "owner", 14),
    ("priority", "prio", 6),
    ("runtime", "runtime", 9),
    ("vram", "vram", 10),
    ("ram", "ram", 10),
    ("eta", "eta", 14),
    ("desc", "description", 36),
]
COLUMN_BY_KEY = {key: (label, width) for key, label, width in COLUMNS}
DEFAULT_COLUMN_ORDER = [key for key, _, _ in COLUMNS]
DEFAULT_COLUMN_WIDTHS = {key: width for key, _, width in COLUMNS}
COLUMN_MIN_WIDTHS = {
    "id": 4,
    "status": 7,
    "node": 10,
    "project": 6,
    "owner": 8,
    "priority": 4,
    "runtime": 6,
    "vram": 6,
    "ram": 6,
    "eta": 7,
    "desc": 10,
}
COLUMN_MAX_WIDTHS = {
    "id": 14,
    "status": 14,
    "node": 60,
    "project": 60,
    "owner": 40,
    "priority": 10,
    "runtime": 14,
    "vram": 16,
    "ram": 16,
    "eta": 24,
    "desc": 100,
}
SORT_KEYS = ["id", "status", "node", "project", "owner", "priority", "runtime", "vram", "ram", "eta"]
TUI_LAYOUT_FILE = sch.STATE_DIR / "tui_layout.json"
HEADER_SEPARATOR = "│"
HEADER_RESIZE_GRAB_CELLS = 2
NODE_CACHE_FRESH_S = max(1, int(os.environ.get("SCHEDULEURM_TUI_NODE_CACHE_FRESH_S", "300")))
NODE_CACHE_FALLBACK_S = max(
    NODE_CACHE_FRESH_S,
    int(os.environ.get("SCHEDULEURM_TUI_NODE_CACHE_FALLBACK_S", "86400")),
)
TASK_LOG_TAIL_LINES = max(1, int(os.environ.get("SCHEDULEURM_TUI_TASK_LOG_LINES", "120")))


class TaskLogScreen(ModalScreen):
    CSS = """
    TaskLogScreen {
        align: center middle;
    }
    #task_log_dialog {
        width: 92%;
        height: 82%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    #task_log_title {
        height: 1;
        color: $accent;
        margin-bottom: 1;
    }
    #task_log_body {
        height: 1fr;
    }
    """
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def __init__(self, title: str, body: str):
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._title, id="task_log_title", markup=False),
            TextArea(
                self._body,
                read_only=True,
                show_line_numbers=False,
                soft_wrap=False,
                id="task_log_body",
            ),
            id="task_log_dialog",
        )

    def action_close(self):
        self.dismiss()


def _read_task_log_payload(task_id: str, lines: int = TASK_LOG_TAIL_LINES) -> dict:
    with sch.state_lock(shared=True, purpose="tui-task-log:snapshot"):
        task, source = sch._find_task_record(task_id, include_archive=True)
    if not task:
        return {
            "ok": False,
            "id": task_id,
            "status": "",
            "node": "",
            "log_path": "",
            "source": "",
            "text": f"task {task_id} not found in queue or archive",
        }
    ok, log_path, text = sch._tail_task_log(task, lines=lines)
    return {
        "ok": bool(ok),
        "id": task.get("id") or task_id,
        "status": task.get("status") or "",
        "node": task.get("node") or "",
        "log_path": log_path or task.get("log_path") or "",
        "source": source,
        "text": text if text else "(empty log)",
    }


def _clamp_column_width(key: str, width) -> int:
    try:
        value = int(width)
    except (TypeError, ValueError):
        value = DEFAULT_COLUMN_WIDTHS.get(key, 10)
    return max(COLUMN_MIN_WIDTHS.get(key, 4), min(COLUMN_MAX_WIDTHS.get(key, 80), value))


def _sanitize_column_order(order) -> list[str]:
    clean = []
    for key in order or []:
        key = str(key)
        if key in COLUMN_BY_KEY and key not in clean:
            clean.append(key)
    for key in DEFAULT_COLUMN_ORDER:
        if key not in clean:
            clean.append(key)
    return clean


def _load_tui_layout() -> tuple[list[str], dict[str, int]]:
    order = list(DEFAULT_COLUMN_ORDER)
    widths = dict(DEFAULT_COLUMN_WIDTHS)
    try:
        raw = json.loads(TUI_LAYOUT_FILE.read_text())
    except Exception:
        return order, widths
    if isinstance(raw, dict):
        order = _sanitize_column_order(raw.get("column_order"))
        raw_widths = raw.get("column_widths") or {}
        if isinstance(raw_widths, dict):
            for key in DEFAULT_COLUMN_ORDER:
                if key in raw_widths:
                    widths[key] = _clamp_column_width(key, raw_widths[key])
    return order, widths


def _save_tui_layout(order: list[str], widths: dict[str, int]) -> None:
    try:
        TUI_LAYOUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "column_order": _sanitize_column_order(order),
            "column_widths": {key: _clamp_column_width(key, widths.get(key)) for key in DEFAULT_COLUMN_ORDER},
        }
        tmp = TUI_LAYOUT_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp, TUI_LAYOUT_FILE)
    except Exception:
        pass


def _header_label(label: str, width: int) -> Text:
    """Render a visible resize handle at each header's right edge."""
    content_width = max(1, int(width))
    if content_width == 1:
        return Text(HEADER_SEPARATOR, style="bright_black")
    visible = str(label)[:content_width - 1].ljust(content_width - 1)
    text = Text(visible)
    text.append(HEADER_SEPARATOR, style="bright_black")
    return text


class SchedulerDataTable(DataTable):
    def on_mouse_down(self, event) -> None:
        handler = getattr(self.app, "_handle_table_mouse_down", None)
        if handler and handler(self, event):
            event.prevent_default()
            event.stop()

    def on_mouse_move(self, event) -> None:
        handler = getattr(self.app, "_handle_table_mouse_move", None)
        if handler and handler(self, event):
            event.prevent_default()
            event.stop()

    def on_mouse_up(self, event) -> None:
        handler = getattr(self.app, "_handle_table_mouse_up", None)
        if handler and handler(self, event):
            event.prevent_default()
            event.stop()


def _configured_probe_names():
    return [
        str(name)
        for name, info in getattr(sch, "NODES", {}).items()
        if not (info or {}).get("monitor_only")
        and not (info or {}).get("retired")
    ]


def _node_probe_cache_snapshot(max_age_s: int):
    """Return a nonblocking, display-ready watcher telemetry snapshot."""
    expected = _configured_probe_names()
    try:
        cached = sch._load_node_probe_cache(max_age_s=max_age_s)
    except Exception:
        cached = {}

    nodes = []
    complete = bool(expected)
    for name in expected:
        rec = cached.get(name) if isinstance(cached, dict) else None
        if isinstance(rec, dict):
            rec = dict(rec)
            # The watcher cache is the TUI's normal telemetry source, not an
            # exceptional fallback.  Snapshot age is rendered separately.
            rec.pop("probe_fallback", None)
            rec.pop("probe_cache_age_s", None)
            nodes.append(rec)
        else:
            complete = False
            nodes.append({
                "name": name,
                "alive": False,
                "probe_pending": True,
                "error": "waiting for first node probe",
            })

    ts = 0.0
    if cached:
        try:
            ts = Path(sch.NODE_PROBE_CACHE_FILE).stat().st_mtime
        except Exception:
            ages = [
                int(rec.get("probe_cache_age_s") or 0)
                for rec in cached.values()
                if isinstance(rec, dict)
            ]
            ts = time.time() - max(ages, default=0)
    return {
        "nodes": nodes,
        "ts": ts,
        "complete": complete,
        "source": "watcher-cache" if cached else "pending",
    }


def _load_display_state():
    """Read atomically-written display state without taking the scheduler lock."""
    try:
        state = sch.load_state()
    except Exception:
        state = {"tasks": []}
    try:
        hist = sch.load_history()
    except Exception:
        hist = {}
    return state, hist


def _probe_snapshot():
    """Gather one node snapshot without ever making the TUI wait on a state lock.

    A healthy watcher already probes and persists every node, so consuming its
    cache avoids duplicate SSH storms.  A live probe is only a recovery path
    for a missing or old cache.  State/history are read after that potentially
    slow operation so its result cannot roll task rows back in time.
    """
    node_snap = _node_probe_cache_snapshot(NODE_CACHE_FRESH_S)
    if not node_snap["complete"]:
        try:
            live_nodes = sch.probe_all()
        except Exception:
            live_nodes = []
        if live_nodes:
            node_snap = {
                "nodes": live_nodes,
                "ts": time.time(),
                "complete": True,
                "source": "live-probe",
            }
        else:
            node_snap = _node_probe_cache_snapshot(NODE_CACHE_FALLBACK_S)

    state, hist = _load_display_state()
    try:
        sch._apply_cpu_slot_accounting_to_nodes(state, node_snap["nodes"])
    except Exception:
        pass
    return {
        "state": state,
        "hist": hist,
        "nodes": node_snap["nodes"],
        "ts": node_snap["ts"],
        "node_source": node_snap["source"],
    }


def _fast_snapshot():
    """Cheap first paint: tasks plus cached/placeholder nodes, with no SSH."""
    state, hist = _load_display_state()
    node_snap = _node_probe_cache_snapshot(NODE_CACHE_FALLBACK_S)
    try:
        sch._apply_cpu_slot_accounting_to_nodes(state, node_snap["nodes"])
    except Exception:
        pass
    return {
        "state": state,
        "hist": hist,
        "nodes": node_snap["nodes"],
        "ts": node_snap["ts"],
        "node_source": node_snap["source"],
    }


def _merge_fast_snapshot(previous: dict, fast: dict) -> dict:
    """Keep newer live telemetry when the watcher cache has not caught up yet."""
    previous = previous or {}
    previous_nodes = previous.get("nodes") or []
    fast_nodes = fast.get("nodes") or []
    previous_ts = float(previous.get("ts") or 0)
    fast_ts = float(fast.get("ts") or 0)
    if previous_nodes and (not fast_nodes or fast_ts < previous_ts):
        fast["nodes"] = previous_nodes
        fast["ts"] = previous_ts
        fast["node_source"] = previous.get("node_source", "live-probe")
    return fast


class SchedulerTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #node_summary { height: auto; min-height: 3; padding: 0 1; color: $text-muted; }
    #filter_input { display: none; height: 3; }
    #filter_input.visible { display: block; }
    DataTable { height: 1fr; }
    """
    BINDINGS = [
        Binding("r", "set_filter('running')", "Running"),
        Binding("q", "set_filter('queued')", "Queued"),
        Binding("a", "set_filter('all')", "All"),
        Binding("f", "toggle_filter", "Filter"),
        Binding("1", "sort_by('id')", ""),
        Binding("2", "sort_by('status')", ""),
        Binding("3", "sort_by('node')", ""),
        Binding("4", "sort_by('project')", ""),
        Binding("5", "sort_by('owner')", ""),
        Binding("6", "sort_by('runtime')", ""),
        Binding("7", "sort_by('vram')", ""),
        Binding("8", "sort_by('ram')", ""),
        Binding("9", "sort_by('eta')", ""),
        Binding("R", "reverse_sort", "Reverse"),
        Binding("p", "bump_priority(1)", "↑prio"),
        Binding("P", "bump_priority(-1)", "↓prio"),
        Binding("l", "view_log", "Log"),
        Binding("c", "copy_id", "Copy ID"),
        Binding("ctrl+r", "refresh_now", "Refresh"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    state_filter = reactive("all")
    sort_key = reactive("id")
    sort_reverse = reactive(False)
    text_filter = reactive("")

    def __init__(self):
        super().__init__()
        self._snap = {"state": {"tasks": []}, "hist": {}, "nodes": [], "ts": 0}
        self._probing = False
        self.column_order, self.column_widths = _load_tui_layout()
        self._table_layout_sig = None
        self._column_drag = None
        self._suppress_header_sort_until = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # markup=False: node_summary contains literal text only (no [bold] etc) and probe error
        # strings can include `[` `]` (e.g. ssh argv `['ssh', '-o', 'BatchMode=yes']`) which Rich
        # would otherwise parse as malformed markup and raise "Expected markup value".
        yield Static("(loading...)", id="node_summary", markup=False)
        yield Input(placeholder="filter (id/project/location/owner/sig/desc) — Enter to apply, Esc to close", id="filter_input")
        yield SchedulerDataTable(id="task_table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self):
        table = self.query_one(DataTable)
        table.show_row_labels = False
        self._apply_table_columns(table)
        # Initial probe + render. Subsequent probes via interval timer.
        # We render only on probe-complete or user action — periodic rendering caused
        # cursor/scroll to snap back to top mid-scroll. Runtime/eta numbers therefore
        # update only every 5s, but the table stays scroll-able.
        self._snap = _fast_snapshot()
        self._render_from_cache()
        self._kick_probe()
        self.set_interval(5.0, self._kick_probe)

    def _layout_signature(self):
        return (
            tuple(self.column_order),
            tuple((key, self.column_widths.get(key)) for key in DEFAULT_COLUMN_ORDER),
        )

    def _apply_table_columns(self, table: DataTable):
        table.clear(columns=True)
        table.show_row_labels = False
        for key in self.column_order:
            label, default_width = COLUMN_BY_KEY[key]
            width = _clamp_column_width(key, self.column_widths.get(key, default_width))
            self.column_widths[key] = width
            table.add_column(_header_label(label, width), key=key, width=width)
        self._table_layout_sig = self._layout_signature()

    def _table_header_virtual_x(self, table: DataTable, event, *, require_header: bool = True):
        if require_header and int(event.y) >= int(getattr(table, "header_height", 1) or 1):
            return None
        scroll_x = int(getattr(table, "scroll_x", 0) or 0)
        row_label_width = int(getattr(table, "_row_label_column_width", 0) or 0)
        virtual_x = int(event.x) + scroll_x - row_label_width
        if virtual_x < 0:
            return None
        return virtual_x

    def _table_header_boundary_hit(self, table: DataTable, event):
        virtual_x = self._table_header_virtual_x(table, event)
        if virtual_x is None:
            return None
        left = 0
        prev = None
        for column in table.ordered_columns:
            key = str(column.key.value)
            width = int(column.get_render_width(table))
            right = left + width
            if prev and abs(virtual_x - left) <= HEADER_RESIZE_GRAB_CELLS:
                return {
                    "key": prev["key"], "left": prev["left"], "right": prev["right"],
                    "width": prev["width"], "virtual_x": virtual_x,
                }
            if abs(virtual_x - right) <= HEADER_RESIZE_GRAB_CELLS:
                return {"key": key, "left": left, "right": right, "width": width, "virtual_x": virtual_x}
            prev = {"key": key, "left": left, "right": right, "width": width}
            left = right
        return None

    def _table_header_hit(self, table: DataTable, event, *, require_header: bool = True):
        virtual_x = self._table_header_virtual_x(table, event, require_header=require_header)
        if virtual_x is None:
            return None
        left = 0
        last = None
        for column in table.ordered_columns:
            key = str(column.key.value)
            width = int(column.get_render_width(table))
            right = left + width
            if left <= virtual_x < right:
                return {"key": key, "left": left, "right": right, "width": width, "virtual_x": virtual_x}
            last = {"key": key, "left": left, "right": right, "width": width, "virtual_x": virtual_x}
            left = right
        if last and virtual_x >= last["right"]:
            last = dict(last)
            last["after_last"] = True
            return last
        return None

    def _event_screen_x(self, event) -> int:
        sx = getattr(event, "screen_x", None)
        return int(sx if sx is not None else event.x)

    def _handle_table_mouse_down(self, table: DataTable, event) -> bool:
        boundary = self._table_header_boundary_hit(table, event)
        hit = boundary or self._table_header_hit(table, event)
        if not hit or hit.get("key") not in COLUMN_BY_KEY:
            self._column_drag = None
            return False
        key = hit["key"]
        screen_x = self._event_screen_x(event)
        if boundary:
            self._column_drag = {
                "mode": "resize",
                "key": key,
                "start_x": screen_x,
                "start_width": int(self.column_widths.get(key, DEFAULT_COLUMN_WIDTHS.get(key, 10))),
                "changed": False,
            }
            try:
                table.capture_mouse()
            except Exception:
                pass
            self._suppress_header_sort_until = time.time() + 0.5
            return True
        self._column_drag = {
            "mode": "reorder",
            "key": key,
            "start_x": screen_x,
            "changed": False,
        }
        try:
            table.capture_mouse()
        except Exception:
            pass
        return False

    def _handle_table_mouse_move(self, table: DataTable, event) -> bool:
        drag = self._column_drag
        if not drag:
            return False
        dx = self._event_screen_x(event) - int(drag.get("start_x") or 0)
        if drag.get("mode") == "resize":
            key = drag["key"]
            new_width = _clamp_column_width(key, int(drag.get("start_width") or 0) + dx)
            if new_width != self.column_widths.get(key):
                self.column_widths[key] = new_width
                drag["changed"] = True
                self._render_from_cache(force_rebuild=True)
                self.title = f"scheduler resize {key}={new_width}"
            return True
        if abs(dx) >= 3:
            drag["changed"] = True
            self._suppress_header_sort_until = time.time() + 0.5
            hit = self._table_header_hit(table, event, require_header=False)
            if hit and hit.get("key"):
                self.title = f"scheduler move {drag['key']} -> {hit['key']}"
            return True
        return False

    def _handle_table_mouse_up(self, table: DataTable, event) -> bool:
        drag = self._column_drag
        if not drag:
            return False
        self._column_drag = None
        try:
            table.release_mouse()
        except Exception:
            pass
        if drag.get("mode") == "resize":
            if drag.get("changed"):
                _save_tui_layout(self.column_order, self.column_widths)
                self.notify(f"{drag['key']} width={self.column_widths.get(drag['key'])}", timeout=2)
            self._suppress_header_sort_until = time.time() + 0.5
            return True
        if not drag.get("changed"):
            return False
        hit = self._table_header_hit(table, event, require_header=False)
        source = drag.get("key")
        target = hit.get("key") if hit else None
        if source == target:
            self._suppress_header_sort_until = time.time() + 0.5
            return True
        if source in self.column_order and target in self.column_order:
            order = [key for key in self.column_order if key != source]
            idx = order.index(target)
            if hit.get("after_last") or hit.get("virtual_x", 0) >= (hit.get("left", 0) + hit.get("right", 0)) / 2:
                idx += 1
            order.insert(idx, source)
            if order != self.column_order:
                self.column_order = order
                _save_tui_layout(self.column_order, self.column_widths)
                self._render_from_cache(force_rebuild=True)
                self.notify(f"moved {source}", timeout=2)
        self._suppress_header_sort_until = time.time() + 0.5
        return True

    # Background probe ---------------------------------------------------
    def _refresh_fast_state(self):
        """Refresh tasks and watcher telemetry while a live probe is in flight."""
        self._snap = _merge_fast_snapshot(self._snap, _fast_snapshot())
        self._render_from_cache()

    def _kick_probe(self):
        if _scheduler_source_changed():
            try:
                self.query_one("#node_summary", Static).update("scheduler.py changed; restarting TUI...")
            except Exception:
                pass
            os.execv(sys.executable, [sys.executable, *sys.argv])
        if self._probing:
            self._refresh_fast_state()
            # Python cannot safely cancel a probe thread.  Starting a new
            # generation here used to discard every result when several down
            # nodes made a probe slower than the stale threshold.
            return
        self._probing = True
        self.run_worker(self._do_probe(), exclusive=False, thread=True, name="probe")

    async def _do_probe(self):
        snap = _probe_snapshot()
        self._snap = snap
        self._probing = False
        self.call_from_thread(self._render_from_cache)

    def on_worker_state_changed(self, event):
        if event.worker.name == "probe" and event.state == WorkerState.ERROR:
            self._probing = False

    # Actions ------------------------------------------------------------
    def action_set_filter(self, status: str):
        self.state_filter = status
        self._render_from_cache()

    def action_sort_by(self, key: str):
        if self.sort_key == key:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_key = key; self.sort_reverse = False
        self._render_from_cache()

    def action_reverse_sort(self):
        self.sort_reverse = not self.sort_reverse
        self._render_from_cache()

    def action_toggle_filter(self):
        inp = self.query_one("#filter_input", Input)
        if inp.has_class("visible"):
            inp.remove_class("visible")
            inp.value = ""; self.text_filter = ""
            self._render_from_cache()
        else:
            inp.add_class("visible")
            inp.focus()

    def on_input_changed(self, event):
        if event.input.id == "filter_input":
            self.text_filter = event.value
            self._render_from_cache()

    def on_input_submitted(self, event):
        if event.input.id == "filter_input":
            self.query_one(DataTable).focus()

    def action_refresh_now(self):
        self._kick_probe()

    def _selected_task_id(self):
        table = self.query_one(DataTable)
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            return row_key.value if row_key else None
        except Exception:
            return None

    def on_data_table_header_selected(self, event):
        """Click a column header → sort by it."""
        if time.time() < self._suppress_header_sort_until:
            return
        self._suppress_header_sort_until = 0.0
        key = str(event.column_key.value) if event.column_key else None
        if key in SORT_KEYS:
            self.action_sort_by(key)

    def action_bump_priority(self, direction: int):
        tid = self._selected_task_id()
        if not tid: return
        import fcntl
        sp = sch.QUEUE_FILE; lp = sch.LOCK_FILE
        with open(lp, "r+") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            state = sch.load_state()
            target = next((t for t in state["tasks"] if t["id"] == tid), None)
            if not target or target.get("status") != "queued":
                fcntl.flock(lf, fcntl.LOCK_UN); return
            same_prio = [t for t in state["tasks"]
                         if t.get("status") == "queued" and t.get("priority") == target.get("priority")]
            same_prio.sort(key=lambda t: t.get("submitted_at") or 0)
            try: idx = same_prio.index(target)
            except ValueError:
                fcntl.flock(lf, fcntl.LOCK_UN); return
            new_idx = idx - direction
            if 0 <= new_idx < len(same_prio) and new_idx != idx:
                neighbor = same_prio[new_idx]
                target["submitted_at"], neighbor["submitted_at"] = neighbor["submitted_at"], target["submitted_at"]
            tmp = sp.with_suffix(".json.tmp"); tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, sp)
            fcntl.flock(lf, fcntl.LOCK_UN)
        # Update cache immediately so render reflects new order.
        self._snap["state"] = state
        self._render_from_cache()

    def action_view_log(self):
        tid = self._selected_task_id()
        if not tid:
            self.notify("no row selected", severity="warning", timeout=2)
            return
        self.notify(f"loading task-log {tid}...", timeout=2)
        self._load_task_log_worker(str(tid))

    @work(thread=True, exclusive=False, group="task-log")
    def _load_task_log_worker(self, tid: str):
        try:
            payload = _read_task_log_payload(tid, lines=TASK_LOG_TAIL_LINES)
        except Exception as e:
            payload = {
                "ok": False,
                "id": tid,
                "status": "",
                "node": "",
                "log_path": "",
                "source": "",
                "text": f"task-log failed: {e}",
            }
        self.call_from_thread(self._show_task_log_payload, payload)

    def _show_task_log_payload(self, payload: dict):
        tid = payload.get("id") or "?"
        status = payload.get("status") or "?"
        node = payload.get("node") or "?"
        log_path = payload.get("log_path") or "?"
        source = payload.get("source") or "?"
        title = f"{tid} {status} {node}:{log_path} [{source}]"
        if not payload.get("ok"):
            self.notify(f"task-log {tid} failed", severity="error", timeout=4)
        self.push_screen(TaskLogScreen(title, payload.get("text") or "(empty log)"))

    def action_copy_id(self):
        """Copy the cursor row's task id to the system clipboard. Tries multiple backends in
        order so it works on WSL2 (clip.exe), X11 (xclip), Wayland (wl-copy), macOS (pbcopy),
        and falls back to Textual's OSC 52 path if available. Notifies via toast either way —
        on failure shows the id so user can mouse-select it from the toast as a last resort."""
        tid = self._selected_task_id()
        if not tid:
            self.notify("no row selected", severity="warning", timeout=2)
            return
        import subprocess
        backends = (
            ["clip.exe"],                            # WSL → Windows clipboard
            ["xclip", "-selection", "clipboard"],    # Linux X11
            ["wl-copy"],                             # Linux Wayland
            ["pbcopy"],                              # macOS
        )
        copied_via = None
        for cmd in backends:
            try:
                subprocess.run(cmd, input=tid.encode(), timeout=2, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                copied_via = cmd[0]; break
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        # Last resort: Textual's OSC 52 path (works in modern terminals like Windows Terminal)
        if not copied_via and hasattr(self, "copy_to_clipboard"):
            try:
                self.copy_to_clipboard(tid); copied_via = "OSC 52"
            except Exception:
                pass
        if copied_via:
            self.notify(f"copied {tid}  (via {copied_via})", timeout=2)
        else:
            self.notify(f"clipboard unavailable — id: {tid}", severity="warning", timeout=4)

    # Render from cache --------------------------------------------------
    def _cell_renderable(self, key: str, value, status: str):
        text = "" if value is None else str(value)
        style = ""
        if status == "running?":
            if key == "status":
                style = "bold yellow"
            elif key == "node":
                style = "yellow"
            else:
                style = "green"
        elif status == "running":
            style = "bold green" if key == "status" else "green"
        return Text(text, style=style, overflow="fold", no_wrap=False)

    def _render_from_cache(self, *, force_rebuild: bool = False):
        try:
            snap = self._snap
            state = snap.get("state", {"tasks": []})
            hist = snap.get("hist", {})
            nodes = snap.get("nodes", [])
            stale = time.time() - snap.get("ts", 0) if snap.get("ts") else None
            stale_tag = "" if stale is None or stale < 7 else f"  (snap {int(stale)}s old)"
            watcher_line = _watcher_status_line()
            summary = _node_summary_line(nodes) + stale_tag
            if watcher_line:
                summary = watcher_line + "\n" + summary
            self.query_one("#node_summary", Static).update(summary)
            tasks = list(state.get("tasks", []))
            if self.state_filter == "running":
                tasks = [t for t in tasks if t.get("status") == "running"]
            elif self.state_filter == "queued":
                tasks = [t for t in tasks if t.get("status") == "queued"]
            else:
                tasks = [t for t in tasks if t.get("status") in ("running", "launching", "queued")]
            tf = (self.text_filter or "").lower().strip()
            if tf:
                def match(t):
                    fields = [sch._format_task_location(t), _display_status(t)]
                    fields.extend(
                        "" if t.get(k) is None else str(t.get(k))
                        for k in ("id", "project", "node", "signature", "description",
                                  "origin", "submitted_by", "process_owner",
                                  "node_probe_state",
                                  "last_probe_unknown_reason", "last_status_sync_reason")
                    )
                    fields.append(sch._format_task_owner(t))
                    return any(tf in f.lower() for f in fields)
                tasks = [t for t in tasks if match(t)]

            now = time.time()
            prio_rank = {"high": 0, "normal": 1, "low": 2}
            def runtime_of(t):
                if t.get("status") == "running" and t.get("started_at"):
                    return now - t["started_at"]
                return 0.0
            def eta_secs(t):
                if t.get("status") in TERMINAL_TASK_STATUSES:
                    return 1e12
                # Sort key for the eta column. Prefer scheduleurm's live
                # eta_seconds, which comes from tqdm/progress log parsing.
                direct = _int_or_default(t.get("eta_seconds"), 0)
                if direct > 0:
                    return direct
                # Fallback to old history EWMA only when the watcher has no ETA.
                # Tasks with no prediction sort to the bottom via 1e12.
                h = hist.get(t.get("signature") or "", {})
                if isinstance(h, int): h = {}
                e = h.get("dur_s_ewma", 0)
                if t.get("status") == "running" and t.get("started_at"):
                    if t.get("auto_adopted") or not e:
                        return 1e12
                    elapsed = now - t["started_at"]
                    if elapsed >= e:
                        return 1e12  # overrun — unknown when it'll finish
                    return e - elapsed
                if t.get("status") == "queued":
                    return e if e else 1e12
                return 1e12
            sortmap = {
                "id": lambda t: t.get("id", ""),
                "status": lambda t: _display_status(t, now),
                "node": lambda t: (
                    t.get("node") or "~",
                    t.get("gpu_idx") if t.get("gpu_idx") is not None else -1,
                ),
                "project": lambda t: t.get("project", ""),
                "owner": lambda t: sch._format_task_owner(t),
                "priority": lambda t: (prio_rank.get(t.get("priority", "normal"), 1), t.get("submitted_at", 0)),
                "runtime": lambda t: -runtime_of(t),
                "vram": lambda t: -int((t.get("current_vram_mb") if t.get("status") == "running" else 0)
                                       or t.get("peak_vram_mb") or t.get("est_vram_mb") or 0),
                "ram": lambda t: -int((t.get("current_ram_mb") if t.get("status") == "running" else 0)
                                      or t.get("peak_ram_mb") or t.get("ram_mb") or 0),
                "eta": lambda t: eta_secs(t),
            }
            keyfn = sortmap.get(self.sort_key, sortmap["id"])
            tasks.sort(key=keyfn, reverse=self.sort_reverse)

            table = self.query_one(DataTable)
            layout_changed = force_rebuild or self._table_layout_sig != self._layout_signature()

            new_id_order = [t["id"] for t in tasks]
            if layout_changed:
                current_id_order = []
            else:
                try:
                    current_id_order = [k.value for k in table.rows.keys()]
                except Exception:
                    current_id_order = []

            def _row_for(t):
                node_str = _display_node(t, now)
                rt = _fmt_min(runtime_of(t)) if t.get("status") == "running" else "-"
                if t.get("status") == "running" and t.get("current_vram_mb"):
                    vram = sch._format_mem_gb(t.get("current_vram_mb", 0))
                else:
                    vram = sch._format_mem_gb(t.get("peak_vram_mb", 0)) if t.get("peak_vram_mb") else (
                        sch._format_mem_gb(t.get("est_vram_mb", 0), approx=True) if t.get("est_vram_mb") else "-")
                # Mirror VRAM column logic for RAM: running=current, terminal=peak, queued=declared.
                if t.get("status") == "running" and t.get("current_ram_mb"):
                    ram = sch._format_mem_gb(t.get("current_ram_mb", 0))
                else:
                    ram = sch._format_mem_gb(t.get("peak_ram_mb", 0)) if t.get("peak_ram_mb") else (
                        sch._format_mem_gb(t.get("ram_mb", 0), approx=True) if t.get("ram_mb") else "-")
                return {
                    "id": t.get("id", "?"),
                    "status": _display_status(t, now),
                    "node": node_str,
                    "project": t.get("project") or "-",
                    "owner": sch._format_task_owner(t),
                    "priority": t.get("priority") or "-",
                    "runtime": rt,
                    "vram": vram,
                    "ram": ram,
                    "eta": _fmt_eta(t, hist),
                    "desc": (t.get("description") or "")[:60],
                }

            if current_id_order == new_id_order and current_id_order:
                # Fast path: same tasks in same order -> update cells in place.
                # Avoids clear() + add_row() which interrupts scroll/cursor.
                for t in tasks:
                    cells = _row_for(t)
                    status = cells.get("status", "-")
                    for col_key in self.column_order:
                        try:
                            table.update_cell(
                                t["id"],
                                col_key,
                                self._cell_renderable(col_key, cells.get(col_key, ""), status),
                                update_width=False,
                            )
                        except Exception:
                            pass
            else:
                # Structural change (sort/filter/task set) → full rebuild with cursor+scroll restore.
                saved_task_id = None
                try:
                    cur_row = table.cursor_row
                    if cur_row is not None and 0 <= cur_row < table.row_count:
                        keys = list(table.rows.keys())
                        if cur_row < len(keys):
                            saved_task_id = keys[cur_row].value
                except Exception:
                    pass
                try:
                    saved_scroll_y = table.scroll_offset.y
                except Exception:
                    saved_scroll_y = 0

                if layout_changed:
                    self._apply_table_columns(table)
                else:
                    table.clear()
                for t in tasks:
                    cells = _row_for(t)
                    status = cells.get("status", "-")
                    table.add_row(
                        *[
                            self._cell_renderable(col_key, cells.get(col_key, ""), status)
                            for col_key in self.column_order
                        ],
                        key=t["id"],
                        height=None,
                    )

                if saved_task_id and saved_task_id in new_id_order:
                    try:
                        table.move_cursor(row=new_id_order.index(saved_task_id), animate=False)
                    except Exception:
                        pass
                try:
                    table.scroll_to(y=saved_scroll_y, animate=False, force=True)
                except Exception:
                    pass
            arrow = "↓" if self.sort_reverse else "↑"
            self.title = (f"scheduler [{self.state_filter}]"
                          + (f" filter={self.text_filter!r}" if self.text_filter else "")
                          + f" sort={self.sort_key}{arrow} | {len(tasks)} tasks"
                          + (" probing..." if self._probing else ""))
        except Exception as e:
            self.query_one("#node_summary", Static).update(f"render error: {e}")


def main():
    SchedulerTUI().run()


if __name__ == "__main__":
    main()
