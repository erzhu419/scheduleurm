"""Linux task memory sampling, shared by remote process and backend probes."""

# Executed by the existing Python 2.7/3 interpreter on each Linux node.
PROCESS_MEMORY_SCRIPT = r'''
def process_pss_kb(pid):
    for filename in ("smaps_rollup", "smaps"):
        try:
            total, found = 0, False
            with open("/proc/%s/%s" % (pid, filename)) as stream:
                for line in stream:
                    if line.startswith("Pss:"):
                        total += int(line.split()[1])
                        found = True
            if found:
                return total
        except (IOError, OSError):
            pass
    return None
'''


TASK_MEMORY_SCRIPT = PROCESS_MEMORY_SCRIPT + r'''
import sys
roots = set(int(value) for value in sys.argv[1:])
parents = {}
for line in sys.stdin:
    fields = line.split()
    if len(fields) >= 5 and fields[4][0] not in ("Z", "X"):
        parents[int(fields[0])] = int(fields[1])
selected = roots.intersection(parents)
frontier = selected.copy()
while frontier:
    children = set(pid for pid, parent in parents.items() if parent in frontier)
    frontier = children.difference(selected)
    selected.update(frontier)
for pid in sorted(selected):
    pss = process_pss_kb(pid)
    if pss is not None:
        sys.stdout.write("%s %s\n" % (pid, pss))
'''
