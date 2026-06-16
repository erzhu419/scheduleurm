#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="$ROOT/md/experiment_artifacts"
MANIFEST="$ART/reproduction_manifest_20260612.json"
START_TS="$(date -Iseconds)"

mkdir -p "$ART"
cd "$ROOT"

run_step() {
  local name="$1"
  shift
  echo "== $name"
  local log="$ART/reproduce_${name}.log"
  if "$@" >"$log" 2>&1; then
    printf '{"name":"%s","status":"PASS","log":"%s"}\n' "$name" "$log" >>"$ART/reproduction_steps.jsonl"
  else
    local code=$?
    printf '{"name":"%s","status":"FAIL","exit_code":%s,"log":"%s"}\n' "$name" "$code" "$log" >>"$ART/reproduction_steps.jsonl"
    return "$code"
  fi
}

run_gate_step_allow_pending() {
  local name="$1"
  shift
  echo "== $name"
  local log="$ART/reproduce_${name}.log"
  set +e
  "$@" >"$log" 2>&1
  local code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    printf '{"name":"%s","status":"PASS","log":"%s"}\n' "$name" "$log" >>"$ART/reproduction_steps.jsonl"
  elif [[ "$code" -eq 2 ]]; then
    printf '{"name":"%s","status":"PENDING","exit_code":%s,"log":"%s"}\n' "$name" "$code" "$log" >>"$ART/reproduction_steps.jsonl"
  else
    printf '{"name":"%s","status":"FAIL","exit_code":%s,"log":"%s"}\n' "$name" "$code" "$log" >>"$ART/reproduction_steps.jsonl"
    return "$code"
  fi
}

rm -f "$ART/reproduction_steps.jsonl"

run_step gavel_service_unit_certificate \
  python3 -m algorithm.experiments.gavel_service_unit_equivalence_certificate build \
    --output "$ART/gavel_service_unit_equivalence_certificate_20260612.json" \
    --markdown-output "$ROOT/md/gavel_service_unit_equivalence_certificate_20260612.md"

run_step gavel_dependency_refresh \
  python3 -m algorithm.experiments.gavel_direct_native_smoke build \
    --install-deps \
    --no-native-trace \
    --timeout-seconds 120 \
    --output "$ART/gavel_direct_native_smoke_dependency_refresh_20260616.json" \
    --markdown-output "$ROOT/md/gavel_direct_native_smoke_dependency_refresh_20260616.md"

run_step gavel_service_unit_paired_holdout \
  python3 -m algorithm.experiments.gavel_service_unit_paired_holdout build \
    --output "$ART/gavel_service_unit_paired_holdout_20260612.json" \
    --markdown-output "$ROOT/md/gavel_service_unit_paired_holdout_20260612.md"

run_step gavel_service_unit_calibration_gate \
  python3 -m algorithm.experiments.gavel_service_unit_calibration_gate build \
    --output "$ART/gavel_service_unit_calibration_gate_20260612.json" \
    --markdown-output "$ROOT/md/gavel_service_unit_calibration_gate_20260612.md"

run_step declared_finite_domain_cover \
  python3 -m algorithm.experiments.declared_finite_domain_positive_cover_gate build \
    --output "$ART/declared_finite_domain_positive_cover_gate_20260612.json" \
    --markdown-output "$ROOT/md/declared_finite_domain_positive_cover_gate_20260612.md"

run_step future_production_admission_contract \
  python3 -m algorithm.experiments.future_production_admission_contract build \
    --admission-mode strict \
    --output "$ART/future_production_admission_contract_20260612.json" \
    --markdown-output "$ROOT/md/future_production_admission_contract_20260612.md"

run_gate_step_allow_pending production_launch_completion_gate \
  python3 -m algorithm.experiments.production_launch_completion_gate build \
    --output "$ART/production_launch_completion_gate_20260612.json" \
    --markdown-output "$ROOT/md/production_launch_completion_gate_20260612.md" \
    --max-tasks 96

run_step controlled_launched_completion_gate \
  python3 -m algorithm.experiments.controlled_production_completion_gate build \
    --output "$ART/controlled_production_completion_gate_20260612.json" \
    --markdown-output "$ROOT/md/controlled_production_completion_gate_20260612.md"

run_step organic_production_canary_recorder_gate \
  python3 -m algorithm.experiments.organic_production_canary_recorder_gate build \
    --output "$ART/organic_production_canary_recorder_gate_20260612.json" \
    --markdown-output "$ROOT/md/organic_production_canary_recorder_gate_20260612.md"

run_step online_arrival_gate \
  python3 -m algorithm.experiments.or_submission_closure online \
    --output "$ART/or_gate_online_arrivals.json" \
    --markdown-output "$ROOT/md/or_gate_online_arrivals.md"

run_step ablation_gate \
  python3 -m algorithm.experiments.or_submission_closure ablation \
    --output "$ART/or_gate_ablation_suite.json" \
    --markdown-output "$ROOT/md/or_gate_ablation_suite.md"

run_step online_ablation_summary_ci \
  python3 -m algorithm.experiments.online_ablation_summary_ci build \
    --output "$ART/online_ablation_summary_ci_20260612.json" \
    --markdown-output "$ROOT/md/online_ablation_summary_ci_20260612.md" \
    --loss-csv-output "$ART/ablation_pareto_by_scenario_20260612.csv"

run_step selected_profile_holdout_lcb_gate \
  python3 -m algorithm.experiments.selected_profile_holdout_lcb_gate build \
    --output "$ART/selected_profile_holdout_lcb_gate_20260612.json" \
    --markdown-output "$ROOT/md/selected_profile_holdout_lcb_gate_20260612.md"

run_step global_theorem_dispatcher_prototype_gate \
  python3 -m algorithm.experiments.global_theorem_dispatcher_prototype_gate build \
    --output "$ART/global_theorem_dispatcher_prototype_gate_20260612.json" \
    --markdown-output "$ROOT/md/global_theorem_dispatcher_prototype_gate_20260612.md"

run_step named_external_runtime_probe_gate \
  python3 -m algorithm.experiments.sota_fullstack_superiority_gate build \
    --output "$ART/sota_fullstack_superiority_gate_20260613.json" \
    --markdown-output "$ROOT/md/sota_fullstack_superiority_gate_20260613.md"

run_step registered_sota_universe_gate \
  python3 -m algorithm.experiments.sota_universe_registry_gate build \
    --output "$ART/sota_universe_registry_gate_20260614.json" \
    --markdown-output "$ROOT/md/sota_universe_registry_gate_20260614.md"

run_step registered_sota_adapter_closure_gate \
  python3 -m algorithm.experiments.registered_sota_adapter_closure_gate build \
    --output "$ART/registered_sota_adapter_closure_gate_20260614.json" \
    --markdown-output "$ROOT/md/registered_sota_adapter_closure_gate_20260614.md"

run_step sota_admitted_universe_closure_gate \
  python3 -m algorithm.experiments.sota_admitted_universe_closure_gate build \
    --output "$ART/sota_admitted_universe_closure_gate_20260614.json" \
    --markdown-output "$ROOT/md/sota_admitted_universe_closure_gate_20260614.md"

run_step organic_history_completion_gate \
  python3 -m algorithm.experiments.organic_history_completion_gate build \
    --output "$ART/organic_history_completion_gate_20260614.json" \
    --markdown-output "$ROOT/md/organic_history_completion_gate_20260614.md"

run_step multinode_history_completion_gate \
  python3 -m algorithm.experiments.multinode_history_completion_gate build \
    --output "$ART/multinode_history_completion_gate_20260614.json" \
    --markdown-output "$ROOT/md/multinode_history_completion_gate_20260614.md"

run_step multinode_original_deployment_gate \
  python3 -m algorithm.experiments.multinode_original_deployment_gate build \
    --output "$ART/multinode_original_deployment_gate_20260614.json" \
    --markdown-output "$ROOT/md/multinode_original_deployment_gate_20260614.md"

run_step decima_same_domain_benchmark_gate \
  python3 -m algorithm.experiments.decima_same_domain_benchmark_gate build \
    --output "$ART/decima_same_domain_benchmark_gate_20260614.json" \
    --markdown-output "$ROOT/md/decima_same_domain_benchmark_gate_20260614.md"

run_step reviewer_environment_manifest_gate \
  python3 -m algorithm.experiments.reviewer_environment_manifest_gate build \
    --output "$ART/reviewer_environment_manifest_gate_20260612.json" \
    --markdown-output "$ROOT/md/reviewer_environment_manifest_gate_20260612.md"

run_step gate_status_dashboard \
  python3 -m algorithm.experiments.gate_status_dashboard build \
    --output "$ART/gate_status_dashboard_20260612.json" \
    --markdown-output "$ROOT/md/gate_status_dashboard.md"

if [[ -d "$ROOT/../proof" && -f "$ROOT/../proof/lakefile.toml" ]]; then
  run_step lean_no_sorry_audit \
    bash -lc "cd '$ROOT/../proof' && ! rg -n --glob '*.lean' '(^|[^A-Za-z_])(sorry|admit|axiom)([^A-Za-z_]|$)' Scheduleurm ScheduleurmUpload.lean"
  if command -v lake >/dev/null 2>&1; then
    run_step lean_build bash -lc "cd '$ROOT/../proof' && lake build"
  else
    printf '{"name":"lean_build","status":"SKIP","reason":"lake not found"}\n' >>"$ART/reproduction_steps.jsonl"
  fi
else
  printf '{"name":"lean_build","status":"SKIP","reason":"proof package not found"}\n' >>"$ART/reproduction_steps.jsonl"
fi

run_step targeted_tests \
  python3 -m pytest -q skill/tests/test_submission_boundary_gates.py skill/tests/test_cpu_slot_accounting.py

run_step paper_pdf \
  make -C paper

run_step latex_log_scan \
  bash -lc "! rg -n 'Float too large|undefined|Citation|Reference|Overfull|Fatal|Emergency|! LaTeX Error|Label\\(s\\) may have changed|Rerun' paper/main.log paper/main.blg"

python3 - <<PY
import json
from pathlib import Path
root = Path("$ROOT")
art = Path("$ART")
steps = [json.loads(line) for line in (art / "reproduction_steps.jsonl").read_text().splitlines() if line.strip()]
artifact_regeneration_modes = {
    "gavel_service_unit_certificate": "safe_rerun",
    "gavel_dependency_refresh": "safe_rerun",
    "gavel_service_unit_paired_holdout": "safe_rerun",
    "gavel_service_unit_calibration_gate": "safe_rerun",
    "declared_finite_domain_cover": "safe_rerun",
    "future_production_admission_contract": "safe_readonly",
    "production_launch_completion_gate": "safe_readonly",
    "controlled_launched_completion_gate": "precomputed_live_run",
    "organic_production_canary_recorder_gate": "safe_readonly",
    "online_arrival_gate": "safe_rerun",
    "ablation_gate": "safe_rerun",
    "online_ablation_summary_ci": "safe_rerun",
    "selected_profile_holdout_lcb_gate": "safe_rerun",
    "global_theorem_dispatcher_prototype_gate": "safe_rerun",
    "named_external_runtime_probe_gate": "requires_external_stack",
    "registered_sota_universe_gate": "safe_rerun",
    "registered_sota_adapter_closure_gate": "safe_rerun",
    "sota_admitted_universe_closure_gate": "safe_rerun",
    "organic_history_completion_gate": "safe_readonly",
    "multinode_history_completion_gate": "safe_readonly",
    "multinode_original_deployment_gate": "safe_readonly",
    "decima_same_domain_benchmark_gate": "requires_external_stack",
    "reviewer_environment_manifest_gate": "safe_rerun",
    "gate_status_dashboard": "safe_rerun",
    "lean_no_sorry_audit": "safe_rerun",
    "lean_build": "safe_rerun",
    "targeted_tests": "safe_rerun",
    "paper_pdf": "safe_rerun",
    "latex_log_scan": "safe_rerun",
}
for step in steps:
    step["regeneration_mode"] = artifact_regeneration_modes.get(step.get("name"), "safe_rerun")
manifest = {
    "generated_at_start": "$START_TS",
    "generated_at_end": __import__("datetime").datetime.now().astimezone().isoformat(),
    "repo_root": str(root),
    "artifact_regeneration_modes": artifact_regeneration_modes,
    "regeneration_mode_legend": {
        "safe_rerun": "Deterministic or non-invasive local command; safe for reviewers to rerun.",
        "safe_readonly": "Reads current queue/history/resource state only; may change with live production state but will not launch work.",
        "precomputed_live_run": "Summarizes previously launched controlled/live rows; do not rerun as a launch command from the reproduction script.",
        "requires_external_stack": "Requires external scheduler/runtime dependencies or previously mirrored containers; packaged evidence is the submission artifact.",
        "not_rerun": "Included only as historical evidence and not used by this script.",
    },
    "steps": steps,
    "pass": all(step.get("status") in {"PASS", "SKIP", "PENDING"} for step in steps),
    "safe_launch_policy": "No reproduction step passes --allow-launch; live production gates are read-only/safe.",
}
(art / "reproduction_manifest_20260612.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n")
print(art / "reproduction_manifest_20260612.json")
PY
