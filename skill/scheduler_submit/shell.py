"""Shell/script inspection helpers for submit policy."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path


def safe_read_text(path: str, max_bytes: int = 512 * 1024) -> str:
    try:
        with open(path, "rb") as handle:
            return handle.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return ""


def script_invocation_from_cmd(cmd: str, cwd: str = ""):
    try:
        toks = shlex.split(cmd or "")
    except Exception:
        return (None, [])
    if not toks:
        return (None, [])
    first = os.path.basename(toks[0])
    script_i = None
    if first in ("bash", "sh", "zsh", "dash"):
        i = 1
        while i < len(toks):
            tok = toks[i]
            if tok in ("-c", "-lc"):
                return (None, [])
            if tok.startswith("-"):
                i += 1
                continue
            script_i = i
            break
    elif first == "env" and len(toks) >= 3 and os.path.basename(toks[1]) in ("bash", "sh", "zsh", "dash"):
        script_i = 2
    elif toks[0].endswith(".sh"):
        script_i = 0
    if script_i is None or script_i >= len(toks):
        return (None, [])
    script = toks[script_i]
    if not os.path.isabs(script):
        script = os.path.abspath(os.path.join(cwd or os.getcwd(), script))
    if not os.path.exists(script) or not os.path.isfile(script):
        return (None, [])
    return (script, toks[script_i + 1:])


def submit_policy_text(cmd: str, cwd: str = "") -> str:
    parts = [cmd or ""]
    script, _ = script_invocation_from_cmd(cmd, cwd)
    if script:
        text = safe_read_text(script)
        if text:
            parts.append(f"\n# scheduleurm-inspected-wrapper: {script}\n{text}")
    return "\n".join(parts)


def shell_split_statements(text: str):
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(";"):
            part = part.strip()
            if part:
                yield part


def shell_expand_simple(value: str, env: dict) -> str:
    value = (value or "").strip()
    if ((value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))):
        value = value[1:-1]
    m = re.fullmatch(r"\$(\d+)", value)
    if m:
        return env.get(m.group(1), "")
    m = re.fullmatch(r"\$\{(\d+):-([^}]*)\}", value)
    if m:
        return env.get(m.group(1), "") or m.group(2)

    def repl(mo):
        name = mo.group(1)
        suffix = mo.group(2)
        val = env.get(name, "")
        if suffix and val.endswith(suffix):
            return val[:-len(suffix)]
        return val

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:%([^}]+))?\}", repl, value)
    value = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda mo: env.get(mo.group(1), ""), value)
    return value


def seed_value_substitute(value, seed: str):
    if value is None:
        return None
    if isinstance(value, list):
        return [seed_value_substitute(v, seed) for v in value]
    text = str(value)
    text = text.replace("${seed}", str(seed))
    text = re.sub(r"\$seed\b", str(seed), text)
    return text


def expand_simple_seed_loop_inner(inner: str):
    if "for seed in" not in (inner or ""):
        return []
    m = re.match(
        r"(?s)^(?P<prefix>.*?)\bfor\s+seed\s+in\s+"
        r"(?P<seeds>[0-9][0-9 \t]*)\s*;\s*do\s+"
        r"(?P<body>.*?)\s*;\s*done\s*$",
        inner.strip(),
    )
    if not m:
        return []
    seeds = [s for s in m.group("seeds").split() if s]
    if len(seeds) <= 1:
        return []
    prefix = m.group("prefix").strip()
    while prefix.endswith(";"):
        prefix = prefix[:-1].rstrip()
    body = m.group("body").strip()
    expanded = []
    for seed in seeds:
        seed_body = seed_value_substitute(body, seed)
        new_inner = "; ".join([p for p in (prefix, seed_body) if p])
        expanded.append((seed, new_inner))
    return expanded


def simple_shell_env_from_script(script_text: str, script_args: list) -> dict:
    env = {str(i + 1): str(v) for i, v in enumerate(script_args or [])}
    for stmt in shell_split_statements(script_text):
        if stmt.startswith(("export ", "local ")):
            stmt = stmt.split(None, 1)[1].strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", stmt)
        if not m:
            continue
        name, val = m.group(1), m.group(2).strip()
        if "$(" in val or "`" in val:
            continue
        env[name] = shell_expand_simple(val, env)
    return env


def extract_script_cd_dir(script_text: str, script_path: str = "", cwd: str = "") -> str:
    for stmt in shell_split_statements(script_text):
        m = re.match(r"^cd\s+(.+)$", stmt)
        if not m:
            continue
        target = shell_expand_simple(m.group(1).strip(), {
            "HOME": str(Path.home()),
            "0": script_path or "",
        })
        if target in ('"$(dirname "$0")"', "'$(dirname \"$0\")'") and script_path:
            return os.path.dirname(script_path)
        if target.startswith("$(dirname"):
            return os.path.dirname(script_path) if script_path else (cwd or os.getcwd())
        if not os.path.isabs(target):
            base = os.path.dirname(script_path) if script_path else (cwd or os.getcwd())
            target = os.path.abspath(os.path.join(base, target))
        return target
    return cwd or (os.path.dirname(script_path) if script_path else os.getcwd())


def arg_value(tokens: list, name: str):
    prefix = name + "="
    for i, tok in enumerate(tokens or []):
        if tok == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


def abs_under(base: str, path: str) -> str:
    if not path:
        return ""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base or os.getcwd(), path))
