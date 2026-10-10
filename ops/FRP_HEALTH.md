# FRP SSH recovery on GPU1 / GPU2

`scheduleurm-frp-health.timer` checks the actual public SSH banner every minute
on each host. Local monitoring does not depend on scheduler or WSL; automatic
restarts now require an independent WSL observation.
The script requires only the host's Python 3.6+ standard library.
`install_frp_health.py` installs the per-host JSON/source/unit bundle supplied as
its base64 JSON argument; it validates FRPC configuration before enabling checks.

An active FRP service is restarted only after three consecutive failed public
checks, each confirmed by a different, fresh external observation. Local SSH and
both clients' FRPS port 7000 checks must be healthy. Missing/stale observations,
external success, SSH throttling, active authenticated SSH sessions, or an
operator-stopped service veto automatic restart. Attempts remain limited to two
per hour, with a 15-minute cooldown. All prerequisites are checked again after
capturing evidence, immediately before any restart.
Recent native FRPC registration errors `port unavailable` and
`send NewProxy request to plugin error` also veto restart: these faults need
provider-side investigation, not a shared client restart. Failed samples retain
the latest registration event and its age. A subsequent registration success or
an event older than 120 seconds clears this veto.

`scheduleurm-frp-observer.timer` runs on WSL once per minute. Fresh socket
handshakes cannot reuse SSH masters. Small reports are delivered over existing
SSH routes; if direct delivery fails, the other GPU's SSH tunnel forwards to the
target's LAN port 22 using its already-known public host key. Both LAN routes
were verified without changing accounts, keys, or FRP. All operations share
scheduler's existing transport quota and do not acquire its state lock.

Runtime paths on each server:

- `/usr/local/lib/scheduleurm/frp_health.py`: installed script.
- `/etc/scheduleurm/frp-health.json`: public endpoint and local evidence paths.
- `/home/researcher/.cache/scheduleurm-frp/external.json`: latest WSL observation
  on each GPU, atomically replaced by its existing user. Root reads this path
  from `external_probe_path`. Observations expire after 120 seconds.
- `/var/lib/scheduleurm-frp-health/latest.json`: last check and decision.
- `/var/lib/scheduleurm-frp-health/health.jsonl*`: minute samples, three 4 MiB files.
- `/var/lib/scheduleurm-frp-health/incident-*.json`: last 12 incidents, captured
  at the first failed check.
- `/var/lib/scheduleurm-frp-health/restart-*.json`: pre-restart snapshots kept
  for seven days, including restart results and subsequent probes. Transient
  incidents cannot evict these records; aborted restart attempts are recorded.
- `/var/log/scheduleurm-frp/frpc.log`: native FRPC debug log, with three-day
  native daily rotation. External copytruncate is removed because it creates
  sparse holes when FRPC continues writing at its previous file offset.

Evidence includes public/local/control connectivity, FRP file descriptors and
logging-pipe occupancy, SSH processes from all users, TCP counters, interface
errors, socket queues/retransmits, and recent SSH/kernel/NetworkManager journals.
Minute samples also include load average and CPU/memory/I/O pressure. Incident
snapshots retain wpa_supplicant disconnect reasons and current wireless status
without scanning, reassociating, or changing power management.
TCP counters are read directly from procfs without seeking. SSH pre-authentication
child counts and the effective admission limits help diagnose older OpenSSH
versions that silently close throttled connections. Minute failure records also
include SSH journal line counts and the reason an independent check was rejected.
Sparse logging tails omit
NUL padding and record its size instead.
Full FRP configuration, credentials, project data, and checkpoints are not copied.

Initial deployment configures native FRPC logging and redirects supervisor output to
journald, without restarting a healthy shared gateway. These logging settings
take effect on the next FRP restart. Check the native log then; the provider's
FRPEMBED supervisor may rewrite its generated configuration.

For installed hosts, `install_frp_health.py --update-policy BASE64_BUNDLE` updates
only the collector source and `external_probe_path`, backs up both, and resets
the failure streak while preserving restart cooldowns. It pauses only the health
timer and lets an in-flight check finish. FRP settings, units and processes are
not changed. The bundle contains `script` and `external_probe_path`.

WSL units live in this repository's `systemd/scheduleurm-frp-observer.*` and run
`ops/frp_observer.py --config ops/frp-observer.json`. Its latest compact report is
`~/.cache/scheduleurm-frp/observer-latest.json`; delivery errors are also in its
user-service journal. The observer does not create persistent monitoring masters.

## Known Limits

Restarting FRP disconnects its existing tunneled connections, including other
users. The watchdog does not restart SSH, networking, the server, or experiments.
Automatic restart is deliberately withheld while authenticated users remain
connected. If WSL is off, both public paths are lost, or the LAN SSH delivery also
fails, monitoring continues but automatic restart waits for external confirmation.
It cannot repair a provider outage or a physical network failure. A successful
SSH banner check detects the observed forwarding failure, not every possible
authentication or downstream-HPC failure. Root-cause confirmation needs a
captured recurrence; Wi-Fi events or high connection counts alone are not proof.
Public FRP connections appear as loopback traffic to SSH, so banning
`127.0.0.1` would also ban legitimate users. This bundle does not change SSH
admission settings, accounts, authentication, firewall rules, or networking.

Stop automatic recovery with `sudo systemctl stop scheduleurm-frp-health.timer`.
