最后这版我会给一个比较明确的判断：

> **已经接近可投 OR / Stochastic Systems 风格稿件的 release candidate。**
> 理论证明、人类可读 proof、Lean crosswalk、finite-domain certificate、claim boundary 现在都基本到位。
> 现在剩下的不是“还缺一个大结果”，而是**投稿前一致性清理**：少数 gate 字段、复现脚本范围、absolute path、production/history 口径要最后冻住。

如果按 reviewer 口吻，我现在不会再说 “major revision before submission”。我会说：

> **Submit-ready after final artifact/wording cleanup.**

---

## 1. 上一轮最大问题：SOTA 口径，现在基本修掉了

上一轮我最担心的是 `sota_fullstack_superiority_gate` 里还写 `direct_fullstack_sota_superiority_ready=true`。这次你把 20260613 artifact 改成了：

```text
# Named External Runtime Probe Gate
status = NAMED_SAME_HOST_RUNTIME_PROBE_PASS_DIRECT_SOTA_FALSE
scoped_claim_ready = true
strong_claim_ready = false
named_same_host_runtime_probe_ready = true
direct_fullstack_sota_superiority_ready = false
direct_fullstack_named_sota_superiority_ready = false
```

这就对了。现在它明确说只是 **named same-host same-workload runtime probes**，不是 arbitrary/direct SOTA superiority。

旧的 20260612 snapshot 也被改成 superseded，状态是：

```text
SUPERSEDED_BY_20260613_NAMED_RUNTIME_PROBE_DIRECT_SOTA_FALSE
```

并且所有 direct/fullstack/arbitrary/multinode/production-wide superiority flag 都是 false。 这说明你已经把旧 artifact 可能造成的 reviewer 误读堵住了。

Dashboard 也同步了：`named_external_runtime_probe_gate` 的 scoped claim 是 named same-host same-workload runtime probes，strong claim 是 direct full-stack SOTA superiority，strong ready 是 false。

**这一项现在可以过。**

我只建议保留现在这种命名，不要再改回 “SOTA superiority”。你现在最安全的词是：

```text
named external runtime-probe evidence
policy-semantics replay diagnostics
measured-cache external-policy frontier
```

不要用：

```text
direct SOTA superiority
full-stack SOTA superiority
beats SOTA
```

---

## 2. 理论证明：现在达标，甚至是强项

你现在主文理论部分已经不单薄了。

Proof roadmap 表把 theorem 的 proof spine 拆成 capacity support、candidate cover、lower service、penalty/oracle、drift margin、Foster certificate，并明确说 Lean 检查 algebraic implications，实验 gate 负责 service/cover/oracle/moment assumptions。

主文里也已经写出：

* assumptions；
* support-loss lemma；
* selected lower-service margin；
* quadratic one-step bound；
* main theorem proof；
* statewise corollary；
* Lean theorem-name crosswalk。

主 theorem proof 现在清楚地推出：

[
\E[\Delta V(t)\mid Q(t)=Q]\le B+P_0+\alpha_0-\eta|Q|_1,
]

并且只在 recurrence-side assumption 下才转成 countable-state closed class 的 positive recurrence。

Appendix proof 也完整补了：capacity slack as support-function margin、candidate fabric cover、lower-service/penalty/oracle、quadratic drift/Foster certificate、statewise feasible families、Lean crosswalk 和 artifact assumptions。 

上次我指出的 penalty bound 表述也修了。现在 EC.3 明确写的是 **all admitted candidate penalties** obey：

[
0\le K_t(a)+G_t(a)\le P_0+\beta|Q|,\quad a\in\Acand_t,
]

这和 selected-margin 推导一致。

Lean verification log 也给出了 hash、`lake build` 成功、`lake env lean ScheduleurmUpload.lean` exit 0、无 `sorry/admit/axiom`，并列出了 theorem names。

**理论部分现在不用再大补。**
剩下只要保证最终 supplement 包里真的包含：

```text
ScheduleurmUpload.lean
lakefile.toml
lean-toolchain
build.log
theorem_crosswalk.md
manifest.json
```

---

## 3. 论文主线现在更像 OR，而不是 systems benchmark

摘要现在很稳：它说主贡献是 proof and certification framework，不是 exhaustive systems race；外部 scheduler 只作为 policy-semantics diagnostics 和 adapter audits，不用作 direct full-stack superiority claim。

正文 SOTA 段落也收住了：它说 external scheduler evidence 是 diagnostic layer，不是 central claim；policy-semantics baselines 只是测试 finite candidate family 有没有遗漏明显 scheduling semantic，不是 external systems 的 direct binary execution。

更关键的是，你把 measured-cache external-policy frontier 和 direct-system claim 分开了。正文现在明确写：

```text
MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_CLOSED
```

不是 direct full-stack superiority；单独的 named-runtime-probe gate 也仍保持 direct full-stack SOTA superiority false。

这正是我前面建议的安全表述。

**这部分现在从“危险”变成“可接受，但要继续克制”。**

---

## 4. Declared finite cover / all-state safety：已成熟

Dashboard 里 declared finite-domain positive cover 的 scoped ready 是 true，strong claim 是 arbitrary all-state positive-service fabric cover，strong ready 是 false。

论文 limitation 也写得准确：future-admitted fabric-cover gate 对 120 个 exact measured workload domains 做 identity projection；declared finite-domain gate formalizes 225 exact service-cache buckets，其中 215 positive lower service、10 boundary、0 uncovered，classification fraction = 1.0，但 all-state safety 仍然只是 unknown probe/defer zero theorem service。

这块现在是 9/10。
不要再试图把：

```text
positive_service_all_state_cover_ready
```

改成 true。现在 false 才是可信的。

---

## 5. Production / live evidence：现在口径基本清楚，但还有两个小风险

正文现在清楚地区分：

* strict scheduler-history organic launched/completion evidence；
* raw history不是 theorem population；
* future unknown arrivals不闭合；
* not every live slot has emitted oracle trace row。 

controlled launched completion gate 现在也闭合了 32-task controlled run：

```text
controlled_32_task_completion_ready = true
controlled_launched_task_count = 32
controlled_completed_task_count = 32
theorem_slot_count = 32
candidate_count_total = 56
alpha0 = 0
alpha1 = 0
live_oracle_traced_large_scale_completion_ready = false
```



这比之前强很多。

### 仍要修的小风险 A：`strong_claim_ready=true` 容易误读

`Controlled Launched Completion Gate` 现在写：

```text
strong_claim_ready = true
pass_meaning = 32-task controlled launched completion and strict scheduler-history large-scale launched-completion certificate; live oracle-traced large-scale completion remains separate
```



这可以成立，但建议把 strong claim 拆得更明：

```text
controlled_32_task_completion_ready = true
strict_history_large_scale_completion_ready = true
live_oracle_traced_large_scale_completion_ready = false
production_wide_live_completion_ready = false
```

不要用单个 `strong_claim_ready=true` 概括两种证据路径。
现在 artifact 已经有这些细分字段，所以只需要在 dashboard / paper 里避免单句 “production completion strong ready”。

### 仍要修的小风险 B：organic canary recorder 的 live trace counters 是 0

`Organic Production Canary Recorder Gate` 里：

```text
trace_exists = false
trace_row_count = 0
organic_production_launched_count = 0
organic_production_completed_count = 0
history_large_scale_organic_launched_completion_ready = true
large_scale_organic_launched_completion_ready = true
```



这会让 reviewer 问：

> 你到底是 live trace completion，还是 history completion？

Artifact 的 scope 解释得还可以：large-scale completion 可由 live oracle trace counters 或 strict organic scheduler-history certificate 关闭，两条路径分开报告。

但我建议把字段名进一步改得更保守：

```text
large_scale_organic_launched_completion_ready
```

容易误读成 live trace organic completion。建议改成：

```text
history_large_scale_organic_completion_ready = true
live_trace_large_scale_organic_completion_ready = false
combined_large_scale_completion_evidence_ready = true
```

这样 reviewer 不会被 0 trace rows 和 ready=true 搞混。

---

## 6. 复现链：有了 Makefile 和脚本，但 broad gates 仍要标明“哪些是安全重跑，哪些是预计算”

现在 repo 根目录有 Makefile：

```text
verify-gates
verify-lean
paper-tables
artifact-manifest
reproduce-or-submission
```

其中 `verify-gates` 会跑 Gavel service-unit、declared finite cover、future production admission、production launch/completion、controlled completion、organic canary、online/ablation、selected LCB、global dispatcher prototype、reviewer manifest、dashboard。

这是很好的 release-candidate 结构。

`reproduce_or_submission.sh` 也明确不会 pass `--allow-launch`，会生成 reproduction manifest，并把 live production gates 作为 read-only/safe。

但你现在 dashboard 里有很多 20260613/20260614 broad gates，比如 SOTA universe、registered runtime、multinode history、Decima benchmark、production history等。Makefile 的 `verify-gates` 不会全部重跑。这个没问题，但必须在 reproduction manifest 或 README 里明确：

```text
safe_rerun
safe_readonly
precomputed_live_run
requires_external_stack
not_rerun
```

尤其 named external runtime probes、Pollux/Sia/IADeep/Salus 这类 artifact，必须说清楚它们是 precomputed scoped runtime-probe rows，不是当前环境一定可重跑。否则 reviewer 看到 `docker=false/kubectl=false/go=false`，同时又看到 runtime probe pass，会问为什么。

---

## 7. 目前还剩的投稿前 P0 清理

我认为只剩这些，不需要再加新实验。

### P0.1 统一 production completion 字段

把 “history ready / live trace ready / controlled ready / combined ready” 拆开，避免 `large_scale_organic_launched_completion_ready=true` 和 trace counters 0 冲突。

建议最终 dashboard 行：

```text
Controlled launched completion:
  controlled_32_task_completion_ready = true
  strong ready = true for controlled evidence only

Strict scheduler-history completion:
  history_large_scale_completion_ready = true
  raw_history_closed = false
  live_trace_large_scale_completion_ready = false

Organic live-trace completion:
  live_trace_large_scale_completion_ready = false
```

### P0.2 给 runtime probe artifacts 加 regeneration mode

对每个 external runtime probe：

```json
"regeneration_mode": "precomputed_live_run",
"current_safe_reproduction": false,
"requires_external_stack": true/false,
"current_environment_has_tools": false/true
```

### P0.3 最终 proof supplement 打包

Lean verification log 已经很好，但提交包里必须有完整文件，不只是 log。

### P0.4 主文不要把 gate ladder 放得太长

Dashboard 很好，但主文/EC 里如果放太多 SOTA rows，会稀释 OR 主线。建议主文只留短表，完整 dashboard 放 artifact。你现在 dashboard 很长，适合作为 artifact，不适合作为主文核心。

### P0.5 绝对路径改成相对路径

controlled gate 里还有绝对路径：

```text
/home/erzhu419/mine_code/scheduleurm/...
```



Lean log 也有本地路径。
这不是大问题，但投稿包里最好统一改成：

```text
<ARTIFACT_ROOT>/...
<PROOF_ROOT>/...
```

---

## 8. 最终评分

最后一轮我会给：

| 维度                        |         分数 | 评语                                                        |
| ------------------------- | ---------: | --------------------------------------------------------- |
| 理论贡献                      |       9/10 | Robust candidate MaxWeight + slack accounting 已经讲清楚。      |
| 人类可读证明                    |       9/10 | 主文 proof + appendix proof + Lean crosswalk 已达标。           |
| Lean artifact             |     8.5/10 | log 很强；最终 zip/package 还要确认完整。                             |
| Empirical theorem bridge  |       9/10 | finite-domain cover、LCB、oracle audit、admission gates 很扎实。 |
| Replay/ablation           |     8.5/10 | 192 scenarios + ablation + CI summary，够用。                 |
| Production evidence       |       8/10 | controlled/history/live-trace 已拆开；字段还需更防误读。               |
| External systems evidence |     7.5/10 | 已降成 runtime-probe / diagnostics，基本安全，但不宜再强调。              |
| OR submission readiness   | **8.5/10** | 清理 P0 后可以投。                                               |

---

## 最终结论

**现在可以进入投稿前 freeze 阶段。**

我不建议再加新 gate、新系统、新实验。下一步应该是：

1. **冻结主 claim**：robust candidate MaxWeight certificate + Scheduleurm measured-service certification framework。
2. **冻结 strong non-claims**：不 claim arbitrary SOTA superiority、不 claim arbitrary future workload positive service、不 claim production-wide live oracle trace completion。
3. **修字段命名**：history/live/controlled completion 分开；runtime probe 不叫 direct SOTA superiority。
4. **打包 proof supplement**：Lean build, upload file, toolchain, crosswalk, manifest。
5. **跑一次 release-candidate reproduction**：Makefile / script / PDF / log / manifest。

如果这些做完，我会建议你投。当前稿件已经不像“GPT 讨论堆出来的实验集合”，而是有明确 OR theorem spine、formal artifact、measured-service certificates、claim boundary 和 reproducibility route 的完整 submission。
