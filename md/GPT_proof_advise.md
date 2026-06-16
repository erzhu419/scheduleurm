是的，**论文里还需要补理论/证明部分**。Lean 证明是很强的 artifact，但对 OR reviewer 来说，**不能把核心证明几乎全放到 Lean 里**。更好的定位是：

> 正文/EC 给出人类可读、自洽的数学证明；Lean 作为形式化验证和 artifact reproducibility。

否则 reviewer 可能会说：我相信你有 Lean 文件，但我没法在论文阅读层面判断 theorem 为什么成立、假设如何被消耗、哪些条件是关键、哪些只是 proof engineering。

你现在摘要已经把理论定位成核心贡献：configuration-action stochastic processing network、robust candidate MaxWeight、explicit slack accounting、finite-set Foster recurrence certificate，并说 fixed-family / statewise variants formalized in Lean。 这意味着理论不是附属品，而是主贡献之一。既然如此，论文里必须把 proof spine 写到 reviewer 能独立跟下来的程度。

---

# 结论

我建议你补：

```text
主文增加 1.5–2.5 页 proof roadmap + main proof sketch
EC / Appendix 增加 6–10 页人类可读证明
Lean supplement 保留为 formal verification
```

不要把所有 Lean 细节搬进论文，但要把 **关键 lemma chain** 写完整。

---

# 为什么现在理论部分会显得单薄

你现在的论文已经有 main theorem 和 theorem-policy crosswalk。crosswalk 说明 theorem-facing rows 暴露 robust MaxWeight score semantics，live hook 只是 integration surface，不是 stability certificate。 这个很好。

但问题是：**证明叙事还不够。** 现在读者能看到 theorem 结果和 artifact 对齐，却看不到足够细的数学桥：

```text
capacity slack
→ support-function slack
→ candidate-cover loss
→ lower-service / penalty / oracle-error loss
→ residual MaxWeight margin
→ quadratic Lyapunov drift
→ finite-set Foster recurrence
```

这条链如果只在 Lean 里，OR reviewer 可能会认为正文“理论黑箱化”。

---

# 你应该在主文里补什么

## 1. 在 Theory section 开头加一张 proof roadmap

放在 Theorem 前或后都可以。建议用一个小表：

| Step               | Statement                                                                       | Consumes slack                |
| ------------------ | ------------------------------------------------------------------------------- | ----------------------------- |
| Capacity support   | Full-action capacity slack implies (q^\top\lambda+\delta|q|*1\le H*{\Afull}(q)) | none                          |
| Candidate cover    | (H_{\Afull}(q)\le H_{\Acand}(q)+L\rho|q|_1)                                     | (L\rho)                       |
| Lower service      | True candidate support is lower bounded by conservative (\underline\mu)         | (\epsilon_{\mathrm{est}})     |
| Penalty/oracle     | Switching, risk, approximate solve                                              | (\beta,\alpha_1,P_0,\alpha_0) |
| Drift              | Residual (\eta>0) gives negative Lyapunov drift outside finite set              | (B)                           |
| Foster certificate | Negative drift outside (|Q|_1\le N)                                             | finite-set recurrence         |

这个表能让 reviewer 一眼明白 theorem 不是一句“Lean proved it”。

---

## 2. 把 theorem proof 拆成 5 个 lemma / proposition

你不需要把它们都写成正式 numbered lemmas，但我建议至少写成 Proposition/Lemma，方便 reviewer 引用。

### Lemma 1：capacity slack implies support slack

正文写：

[
\lambda_i+\delta\le \sum_{a\in\Afull}x_a\mu_i(a)
\quad\forall i
]

则对任意 (q\ge0)：

[
q^\top\lambda+\delta|q|*1
\le
\sum_a x_a q^\top\mu(a)
\le
\max*{a\in\Afull}q^\top\mu(a).
]

这个证明很短，应该放主文。它是整篇 proof 的“经济学含义”：capacity slack 变成 support-function slack。

### Lemma 2：fabric cover support loss

如果 (\Acand) 是 (\rho)-cover，且

[
|\mu(a)-\mu(a')|*\infty \le L d*\Phi(a,a'),
]

那么：

[
H_{\Afull}(q)
\le
H_{\Acand}(q)+L\rho|q|_1.
]

证明也很短：对 full optimum (a^\star)，取 projection (\pi(a^\star)\in\Acand)，然后用 Lipschitz bound。

这部分要放主文，因为它是你“没有偷换 action space”的核心卖点。

### Lemma 3：robust lower-service and oracle loss

定义 theorem score：

[
\Psi_t(a)=Q^\top \underline\mu_t(a)-K_t(a)-G_t(a).
]

approximate oracle 条件：

[
\max_{a\in\Acand_t}\Psi_t(a)-\Psi_t(a_t)
\le
\alpha_0+\alpha_1|Q|_1.
]

penalty 条件：

[
0\le K_t(a_t)+G_t(a_t)\le P_0+\beta|Q|_1.
]

lower-service support error：

[
H_{\Acand}(Q)
\le
\max_{a\in\Acand}Q^\top\underline\mu(a)
+\epsilon_{\mathrm{est}}|Q|_1.
]

推出：

[
Q^\top \underline\mu(a_t)
\ge
Q^\top\lambda
+
\eta|Q|_1
---------

(P_0+\alpha_0),
]

其中：

[
\eta=\delta-(L\rho+\epsilon_{\mathrm{est}}+\beta+\alpha_1).
]

这个 lemma 是 theorem 最关键的一步。现在如果只在 Lean 里，reader 会觉得 theorem 像 magic。建议主文至少给这条不等式的推导。

### Lemma 4：quadratic drift inequality

队列更新：

[
Q_i(t+1)=[Q_i(t)-S_i(t)]^+ + A_i(t).
]

对

[
V(Q)=\frac12\sum_i Q_i^2
]

用标准 inequality：

[
V(Q(t+1))-V(Q(t))
\le
\frac12\sum_i(A_i(t)^2+S_i(t)^2)
+
Q^\top A(t)-Q^\top S(t).
]

再取 conditional expectation，代入：

[
\E[A_i\mid Q]\le \lambda_i,\qquad
\underline\mu_i(a_t)\le \E[S_i\mid Q],
]

得到：

[
\E[\Delta V\mid Q]
\le
B+P_0+\alpha_0-\eta|Q|_1.
]

这一步是 OR/queueing reviewer 最熟悉的部分，必须写清楚。

### Lemma 5 / Corollary：finite-set Foster recurrence

如果

[
B+P_0+\alpha_0+\gamma\le \eta N,
]

则当 (|Q|_1>N) 时：

[
\E[\Delta V\mid Q]\le -\gamma.
]

所以得到 finite-set Foster recurrence certificate。

这里要非常明确：你不是无条件 claim 全 Markov chain positive recurrence，而是 finite-set recurrence certificate；若要转成传统 positive recurrence，还需要 irreducibility / closed communicating class 条件。这个边界非常重要。

---

# 主文 proof sketch 可以这样写

你可以直接加一个 subsection：

```latex
\subsection{Proof idea and slack accounting}
```

结构如下：

```latex
The proof has five inequalities. First, full-action capacity slack is converted to
support slack. Second, the fabric cover transfers the support function from the
full action family to the candidate family with loss L rho. Third, conservative
lower service, penalties, and approximate optimization consume additional slack.
Fourth, the selected action therefore provides queue-weighted service exceeding
arrival load by eta ||Q||_1 up to additive constants. Fifth, the standard quadratic
Lyapunov inequality converts this deterministic margin into conditional drift.
```

然后逐条写上述公式。

主文不用写特别长，但这 5 个 inequality 必须出现。

---

# EC / Appendix 应该补什么

主文写 proof sketch，EC 写完整 proof。

我建议 EC 结构：

```text
EC.1 Capacity and support functions
  Lemma EC.1 downward capacity slack -> support slack

EC.2 Candidate fabric cover
  Lemma EC.2 support loss under rho-cover
  Lemma EC.3 statewise indexed cover version

EC.3 Robust candidate MaxWeight drift
  Proposition EC.4 exact oracle
  Proposition EC.5 approximate oracle

EC.4 Stochastic queue model
  Lemma EC.6 quadratic drift
  Theorem EC.7 finite-support stochastic certificate

EC.5 Lean correspondence
  Table EC.1 paper theorem -> Lean theorem
  Table EC.2 assumptions not discharged by Lean but certified by artifacts
```

这不需要很长。6–10 页足够。

---

# Lean 应该怎么在论文里定位

Lean 不是 proof replacement，而是 proof audit。

建议写：

> The electronic companion gives a conventional proof of the main theorem. The Lean artifact formalizes the same proof spine and its fixed-family, statewise, bounded-second-moment, and approximate-oracle variants. The formalization checks the algebraic slack accounting and recurrence certificate; empirical assumptions such as service lower bounds, fabric cover constants, and oracle gaps are certified by the experiment gates rather than proved by Lean.

这句话非常重要，因为它告诉 reviewer：

```text
Lean proves math implication.
Artifacts certify assumptions.
Paper explains both.
```

否则 reviewer 可能问：Lean 到底证明了实验 claim 了吗？答案是没有，也不应该说有。

---

# 需要不要把 Lean theorem 名字放主文？

主文不用放太多，但 EC 必须有 crosswalk。

建议表格：

| Paper result      | Lean theorem                                                                                         | What it proves                    | Empirical inputs                                |
| ----------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------- |
| Theorem 1 exact   | `main_theorem_robust_candidate_maxweight_stability_under_fabric_cover`                               | exact-oracle drift and recurrence | (\delta,L,\rho,\epsilon_{\mathrm{est}},\beta,B) |
| Theorem 1 approx  | `main_theorem_robust_candidate_maxweight_stability_with_second_moment_bound_approx_oracle`           | approximate oracle version        | (\alpha_0,\alpha_1)                             |
| Statewise version | `main_statewise_calibrated_fabric_robust_candidate_stability_with_second_moment_bound_approx_oracle` | state-dependent feasible family   | per-state gate certificates                     |

这样 reviewer 会知道你不是把 Lean 当装饰，而是精确对应 theorem。

---

# 哪些证明不要放主文

不要把这些放主文展开太多：

1. active-bucket hidden-regime certificate；
2. adaptive sampler detector probability model；
3. Gavel adapter / SOTA gate correctness；
4. production history gate correctness；
5. all artifact JSON schema validation。

这些放 appendix / artifact README 即可。主文理论只服务 main theorem。

---

# 当前论文里最好删减或迁移的部分

你现在 experiments / SOTA / production 细节已经很多。为了给 theory proof 腾空间，建议把以下内容压缩：

1. SOTA action-union gate 的详细解释可以移到 EC。
2. corner-case marginal service probes 两张表可以只留一张主表，另一张去 EC。
3. production live evidence 那段现在很长，可以压成一个表：controlled / shadow / history / live-trace 四列。
4. Claim matrix 主文用短版，完整 gate matrix 去 EC。

腾出来的 2 页给 proof sketch，值很多。

---

# OR reviewer 的心理

OR reviewer 通常不会接受：

> The proof is formalized in Lean; see supplement.

除非正文里也有足够完整的 proof idea。Lean 对他们是 bonus，不是主阅读路径。

更合适的是：

> The proof follows a standard MaxWeight drift structure, but the novelty is the slack accounting through candidate cover, lower-service estimation, penalties, and approximate oracle error. We provide the full human-readable proof in EC and a Lean formalization as a machine-checkable audit.

这句话会让 reviewer 很舒服。

---

# 具体改动清单

## 必须补进主文

1. `Proof roadmap` 表。
2. `capacity slack -> support slack` 推导。
3. `candidate cover -> \(L\rho\)` 推导。
4. `lower-service + penalty + oracle -> residual margin \(\eta\)` 推导。
5. `quadratic Lyapunov drift` 推导。
6. `finite-set Foster certificate` 解释。
7. 一句 Lean 只是 formal audit，不替代 artifact assumptions。

## 必须补进 EC

1. 完整 main theorem proof。
2. approximate oracle proof。
3. statewise feasible family proof sketch。
4. paper theorem ↔ Lean theorem crosswalk。
5. artifact assumptions table。

## 证明包必须补

1. `lakefile.toml`
2. `lean-toolchain`
3. `ScheduleurmUpload.lean` 或明确说明 modular build 版本取代 upload 文件
4. build log
5. manifest
6. SHA-256
7. no-sorry/admit/axiom grep report

---

# 我建议的最终理论结构

论文主文：

```text
3. Robust Candidate MaxWeight
  3.1 Policy
  3.2 Main theorem
  3.3 Proof idea and slack accounting   <-- 新增，1.5–2.5页
  3.4 Formalization and artifact boundary
```

EC：

```text
EC.1 Full proof of Theorem 1
EC.2 Statewise candidate-family extension
EC.3 Approximate-oracle and bounded-second-moment variants
EC.4 Lean theorem crosswalk
EC.5 Artifact assumptions and certificates
```

---

# 最终建议

**要补，而且必须补。**
不是因为 Lean 不强，而是因为 OR 论文要让 reviewer 在不运行 Lean 的情况下也能判断理论贡献。

现在你的理论部分如果保持“一笔带过 + Lean supplement”，风险是：

```text
The theorem may be correct, but the paper does not explain the proof enough.
```

补完上述 proof sketch + EC 后，风险会变成：

```text
The theory is understandable; Lean strengthens confidence.
```

这两者在审稿里差别很大。
