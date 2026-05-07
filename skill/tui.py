"""Interactive TUI for `scheduler status` — sortable table, filter, auto-refresh.
Run via `python ~/.claude/skills/scheduler/scheduler.py tui`.

Probe runs in a background thread so SSH timeouts (up to 5s/node) never block the UI.
Sort/filter operate on the cached snapshot — instant response.

Keys:
  r / q / a    → filter to running / queued / all active
  f            → focus filter input (substring match against id/project/node/sig/desc)
  1..8         → sort by column (id / status / node / project / runtime / vram / ram / eta)
  R            → reverse sort direction
  p / P        → bump task priority up / down (only for queued tasks)
  c            → copy current row's task id to clipboard (paste e.g. into `cancel`/`show`)
  ctrl+r       → force refresh now
  ctrl+c       → quit
You can also CLICK a column header to sort by it (click again = reverse).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.reactive import reactive
    from textual.widgets import DataTable, Footer, Header, Input, Static
    from textual.worker import Worker, WorkerState
except ImportError:
    sys.exit("textual not installed. Run: pip install --user textual")

sys.path.insert(0, str(Path(__file__).parent))
import scheduler as sch  # noqa: E402


def _fmt_min(secs):
    if secs is None or secs < 0: return "-"
    if secs < 60: return f"{int(secs)}s"
    if secs < 3600: return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


def _fmt_eta(t, hist):
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
        gpus = "  ".join(
            f"GPU{g['idx']}={g['used_mb']}/{g['total_mb']}MB({g['used_mb']*100//g['total_mb']}%mem,{g['util_pct']}%util)"
            for g in n["gpus"]
        )
        load = n.get("loadavg")
        load_s = f"load {load:.1f}" if isinstance(load, (int, float)) else ""
        ram_free = n.get("free_ram_mb")
        ram_s = f"ram_free={ram_free}MB" if ram_free is not None else ""
        cpu_s = f"cpu={n.get('free_cpu','?')}/{n.get('total_cpu','?')}"
        tail = "  ".join(s for s in (cpu_s, load_s, ram_s) if s)
        lines.append(f"{n['name']:<11s} {gpus}  {tail}")
    return "\n".join(lines)


COLUMNS = [
    ("id", "id", 6),
    ("status", "status", 9),
    ("node", "node:gpu", 16),
    ("project", "project", 14),
    ("priority", "prio", 6),
    ("runtime", "runtime", 9),
    ("vram", "peak_vram", 10),
    ("ram", "peak_ram", 10),
    ("eta", "eta", 14),
    ("desc", "description", 60),
]
SORT_KEYS = ["id", "status", "node", "project", "priority", "runtime", "vram", "ram", "eta"]


def _probe_snapshot():
    """Background worker — gathers everything the UI needs in one go."""
    state = sch.load_state()
    hist = sch.load_history()
    try:
        nodes = sch.probe_all()
    except Exception:
        nodes = []
    return {"state": state, "hist": hist, "nodes": nodes, "ts": time.time()}


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
        Binding("5", "sort_by('runtime')", ""),
        Binding("6", "sort_by('vram')", ""),
        Binding("7", "sort_by('ram')", ""),
        Binding("8", "sort_by('eta')", ""),
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # markup=False: node_summary contains literal text only (no [bold] etc) and probe error
        # strings can include `[` `]` (e.g. ssh argv `['ssh', '-o', 'BatchMode=yes']`) which Rich
        # would otherwise parse as malformed markup and raise "Expected markup value".
        yield Static("(loading...)", id="node_summary", markup=False)
        yield Input(placeholder="filter (id/project/node/sig/desc) — Enter to apply, Esc to close", id="filter_input")
        yield DataTable(id="task_table", zebra_stripes=True, cursor_type="row")
        yield Footer()

    def on_mount(self):
        table = self.query_one(DataTable)
        for key, label, width in COLUMNS:
            # Don't pin width — let textual auto-fit so click areas are larger.
            table.add_column(label, key=key)
        # Initial probe + render. Subsequent probes via interval timer.
        # We render only on probe-complete or user action — periodic rendering caused
        # cursor/scroll to snap back to top mid-scroll. Runtime/eta numbers therefore
        # update only every 5s, but the table stays scroll-able.
        self._kick_probe()
        self.set_interval(5.0, self._kick_probe)

    # Background probe ---------------------------------------------------
    def _kick_probe(self):
        if self._probing: return
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
        import fcntl, os
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
    def _render_from_cache(self):
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
                    return any(tf in (str(t.get(k, "")) or "").lower()
                               for k in ("id", "project", "node", "signature", "description"))
                tasks = [t for t in tasks if match(t)]

            now = time.time()
            prio_rank = {"high": 0, "normal": 1, "low": 2}
            def runtime_of(t):
                if t.get("status") == "running" and t.get("started_at"):
                    return now - t["started_at"]
                return 0.0
            def eta_secs(t):
                # Sort key for the eta column. Tasks with a clean predicted remaining
                # come first (smallest remaining = soonest). Overrun (elapsed > expected)
                # and auto-adopted (no reliable prediction) sort to the bottom via 1e12.
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
                "node": lambda t: (t.get("node") or "~", t.get("gpu_idx") if t.get("gpu_idx") is not None else -1),
                "project": lambda t: t.get("project", ""),
                "priority": lambda t: (prio_rank.get(t.get("priority", "normal"), 1), t.get("submitted_at", 0)),
                "runtime": lambda t: -runtime_of(t),
                "vram": lambda t: -int(t.get("peak_vram_mb") or 0),
                "ram": lambda t: -int(t.get("peak_ram_mb") or 0),
                "eta": lambda t: eta_secs(t),
            }
            keyfn = sortmap.get(self.sort_key, sortmap["id"])
            tasks.sort(key=keyfn, reverse=self.sort_reverse)

            table = self.query_one(DataTable)

            new_id_order = [t["id"] for t in tasks]
            try:
                current_id_order = [k.value for k in table.rows.keys()]
            except Exception:
                current_id_order = []

            def _row_for(t):
                node_str = (f"{t.get('node','-')}:GPU{t.get('gpu_idx')}"
                            if t.get("node") and t.get("gpu_idx") is not None else (t.get("node") or "-"))
                rt = _fmt_min(runtime_of(t)) if t.get("status") == "running" else "-"
                vram = f"{t.get('peak_vram_mb', 0)}MB" if t.get("peak_vram_mb") else (
                    f"~{t.get('est_vram_mb', 0)}MB" if t.get("est_vram_mb") else "-")
                # Mirror VRAM column logic for RAM: prefer measured peak, fall back to declared.
                ram = f"{t.get('peak_ram_mb', 0)}MB" if t.get("peak_ram_mb") else (
                    f"~{t.get('ram_mb', 0)}MB" if t.get("ram_mb") else "-")
                return {
                    "id": t.get("id", "?"),
                    "status": t.get("status", "-"),
                    "node": node_str,
                    "project": t.get("project") or "-",
                    "priority": t.get("priority") or "-",
                    "runtime": rt,
                    "vram": vram,
                    "ram": ram,
                    "eta": _fmt_eta(t, hist),
                    "desc": (t.get("description") or "")[:60],
                }

            if current_id_order == new_id_order and current_id_order:
                # Fast path: same tasks in same order → update only mutable cells in place.
                # Avoids clear() + add_row() which interrupts scroll/cursor.
                for t in tasks:
                    cells = _row_for(t)
                    for col_key in ("status", "node", "runtime", "vram", "ram", "eta", "priority"):
                        try:
                            table.update_cell(t["id"], col_key, cells[col_key], update_width=False)
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

                table.clear()
                for t in tasks:
                    cells = _row_for(t)
                    table.add_row(cells["id"], cells["status"], cells["node"], cells["project"],
                                  cells["priority"], cells["runtime"], cells["vram"], cells["ram"],
                                  cells["eta"], cells["desc"], key=t["id"])

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
