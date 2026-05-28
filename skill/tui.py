"""Interactive TUI for `scheduler status` — sortable table, filter, auto-refresh.
Run via `python ~/.claude/skills/scheduler/scheduler.py tui`.

Probe runs in a background thread so SSH timeouts (up to 5s/node) never block the UI.
Sort/filter operate on the cached snapshot — instant response.

Keys:
  r / q / a    → filter to running / queued / all active
  f            → focus filter input (substring match against id/project/location/owner/slurm/sig/desc)
  1..9         → sort by column (id / status / node / project / owner / runtime / vram / ram / eta)
  R            → reverse sort direction
  p / P        → bump task priority up / down (only for queued tasks)
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
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.reactive import reactive
    from textual.widgets import DataTable, Footer, Header, Input, Static
    from textual.worker import Worker, WorkerState
except ImportError:
    sys.exit("textual not installed. Run: pip install --user textual")

sys.path.insert(0, str(Path(__file__).parent))
import scheduler as sch  # noqa: E402

_SCHED_SOURCE = Path(getattr(sch, "__file__", Path(__file__).with_name("scheduler.py"))).resolve()
_TUI_SOURCE = Path(__file__).resolve()
_SOURCE_MTIMES = {}
for _src in (_SCHED_SOURCE, _TUI_SOURCE):
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


def _int_or_default(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_eta(t, hist):
    eta = _int_or_default(t.get("eta_seconds"), 0)
    if eta > 0:
        tag = sch._eta_source_tag(t.get("eta_source")) if hasattr(sch, "_eta_source_tag") else (t.get("eta_source") or "?")
        return f"~{_fmt_min(eta)} {tag}"
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


def _node_summary_line(nodes):
    if not nodes: return "(probe pending...)"
    lines = []
    for n in nodes:
        if not n.get("alive"):
            # Defense in depth: error strings often contain ssh argv like ['ssh', '-o', ...]
            # which Rich parses as markup tags → "Expected markup value (...)" render error.
            # Strip the brackets here even though we also disable markup on the Static widget.
            err = (n.get("error", "?") or "?")[:60].replace("[", "(").replace("]", ")")
            lines.append(f"{n['name']:<11s} DOWN ({err})")
            continue
        if n.get("slurm_cluster"):
            lines.append(f"{n['name']:<11s} {sch._format_slurm_cluster_summary(n)}")
            continue
        # Phase 3.3: for `local` (WSL2), supplement NVML util with the DXGI
        # Compute-engine reading (matches Task Manager). RAM display uses
        # scheduler's effective free value so WSL's inflated VM-internal
        # MemAvailable does not distract from real placement capacity.
        def _gpu_segment(g):
            mem_pct = g['used_mb'] * 100 // max(g['total_mb'], 1)
            util = f"{g['util_pct']}%util"
            cu = g.get("util_pct_compute")
            if cu is not None:
                util = f"{g['util_pct']}/{cu}%util(nvml/compute)"
            return (
                f"GPU{g['idx']}={sch._format_mem_gb(g.get('used_mb', 0))}/"
                f"{sch._format_mem_gb(g.get('total_mb', 0))}"
                f"(free={sch._format_mem_gb(g.get('free_mb', 0))},{mem_pct}%mem,{util})"
            )
        gpus = "  ".join(_gpu_segment(g) for g in n["gpus"])
        load = n.get("loadavg")
        load_s = f"load {load:.1f}" if isinstance(load, (int, float)) else ""
        host_cpu = n.get("host_cpu_load_pct")
        if host_cpu is not None:
            wsl_load = n.get("wsl_loadavg")
            if isinstance(wsl_load, (int, float)):
                load_s = f"wsl_load {wsl_load:.1f}, host_cpu {int(host_cpu)}%"
            else:
                load_s = f"host_cpu {int(host_cpu)}%"
        if n.get("probe_fallback"):
            load_s = (load_s + ", " if load_s else "") + str(n.get("probe_fallback"))
        ram_s = sch._format_node_ram_summary(n)
        cpu_s = f"cpu={n.get('free_cpu','?')}/{n.get('total_cpu','?')}"
        claim_s = sch._format_node_claim_summary(n)
        claim_s = claim_s.strip() if claim_s else ""
        tail = "  ".join(s for s in (cpu_s, load_s, ram_s, claim_s) if s)
        lines.append(f"{n['name']:<11s} {gpus}  {tail}")
    return "\n".join(lines)


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


def _probe_snapshot():
    """Background worker — gathers everything the UI needs in one go.

    The TUI is a read-only monitor. Do not take scheduler's global state lock
    here: a watcher dispatch can legitimately hold it while SSH/Slurm work is
    in flight, and blocking on that lock makes the device list look frozen.
    queue.json is written with atomic os.replace(), so unlocked reads are safe
    for display; the watcher remains responsible for mutating/reconciling state.
    """
    try:
        state = sch.load_state()
    except Exception:
        state = {"tasks": []}
    try:
        hist = sch.load_history()
    except Exception:
        hist = {}
    try:
        nodes = sch.probe_all()
    except Exception:
        nodes = []
    return {"state": state, "hist": hist, "nodes": nodes, "ts": time.time()}


def _fast_snapshot():
    """Cheap first paint: show queue rows before SSH/Slurm/node probes finish."""
    try:
        state = sch.load_state()
    except Exception:
        state = {"tasks": []}
    try:
        hist = sch.load_history()
    except Exception:
        hist = {}
    return {"state": state, "hist": hist, "nodes": [], "ts": 0}


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
        yield Input(placeholder="filter (id/project/location/owner/slurm/sig/desc) — Enter to apply, Esc to close", id="filter_input")
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
    def _kick_probe(self):
        if self._probing: return
        if _scheduler_source_changed():
            try:
                self.query_one("#node_summary", Static).update("scheduler.py changed; restarting TUI...")
            except Exception:
                pass
            os.execv(sys.executable, [sys.executable, *sys.argv])
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

    def on_data_table_header_selected(self, event):
        """Click a column header → sort by it."""
        if time.time() < self._suppress_header_sort_until:
            return
        self._suppress_header_sort_until = 0.0
        key = str(event.column_key.value) if event.column_key else None
        if key in SORT_KEYS:
            self.action_sort_by(key)

    def action_bump_priority(self, direction: int):
        table = self.query_one(DataTable)
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            tid = row_key.value if row_key else None
        except Exception:
            return
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

    def action_copy_id(self):
        """Copy the cursor row's task id to the system clipboard. Tries multiple backends in
        order so it works on WSL2 (clip.exe), X11 (xclip), Wayland (wl-copy), macOS (pbcopy),
        and falls back to Textual's OSC 52 path if available. Notifies via toast either way —
        on failure shows the id so user can mouse-select it from the toast as a last resort."""
        table = self.query_one(DataTable)
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            tid = row_key.value if row_key else None
        except Exception:
            tid = None
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
        if status == "running":
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
            self.query_one("#node_summary", Static).update(_node_summary_line(nodes) + stale_tag)
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
                    fields = [sch._format_task_location(t)]
                    fields.extend(
                        "" if t.get(k) is None else str(t.get(k))
                        for k in ("id", "project", "node", "signature", "description",
                                  "origin", "submitted_by", "process_owner",
                                  "slurm_job_id", "slurm_state")
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
                "status": lambda t: t.get("status", ""),
                "node": lambda t: (
                    t.get("node") or "~",
                    0 if t.get("slurm_job_id") else 1,
                    _int_or_default(t.get("slurm_job_id")),
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
                node_str = sch._format_task_location(t)
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
                    "status": t.get("status", "-"),
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
