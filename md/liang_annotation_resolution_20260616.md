# Liang Supervisor Annotation Resolution Log

This log records the non-formula fixes made for Liang's annotated
`paper/main_revise_supervisor.pdf`.  Rows follow the PDF annotation order.
Formula-width and line-breaking comments are recorded but were not changed
because the user explicitly excluded those issues: the Operations Research
template constrains display-equation width and sometimes requires line breaks.

| Page | Annotation | Highlight or nearest text/table | Before | After | Status |
|---:|---|---|---|---|---|
| 1 | “这样表达对么？” | Abstract sentence around “proof and certification framework” | Abstract framed the contribution as a proof/certification framework. | Abstract now states the stochastic network model, robust candidate MaxWeight certificate, Lean formalization, finite-slice Scheduleurm evidence, and quantitative results. | Fixed |
| 1 | “思维过程、中间过程的残留痕迹…摘要里面需要强化理论模型算法和实验…” | Abstract sentence around external scheduler comparisons and adapter evidence | Abstract contained process language such as claim boundaries, adapter checks, and residual audit wording. | Abstract rewritten with sharper model-theory-algorithm-experiment contribution and explicit finite-slice scope. | Fixed |
| 1 | “系统手术式扫描全文的数学符号、公式、证明过程…” | Title-page and first-page manuscript scope | Main text mixed proof, engineering process, and evidence-gate language. | Notation table rewritten one-symbol-per-row; claim matrix, proof roadmap, implementation, experiments, discussion, conclusion, code availability, and references were scanned and revised. | Fixed |
| 2 | “小灰字体都要清楚掉…全文所有图片系统清理” | Figure 1 grey note under panel B | Figure contained small grey explanatory notes. | Figure 1 regenerated without grey footnotes; caption defines the measured-service action table. | Fixed |
| 2 | “清理干净！” | Figure 1 grey note under panel C | Figure contained small process notes under the slack panel. | Panel C simplified to a compact slack-test diagram; grey notes removed. | Fixed |
| 2 | “图片不要花里胡哨…不要流水账” | Figure 1 overall | Figure read like a workflow log with status text. | Figure 1 now has three compact panels: quadrant problem, measured-service action table, and slack test to theorem-facing claims. | Fixed |
| 2 | Figure abbreviation cleanup from the same figure-quality request | Figure 1 labels | Figure labels used unexplained abbreviations such as CNN/LLM, SRPT, and SOTA before their first textual definitions. | Figure labels now use “vision/language kernels,” “short-job/overhead,” and “future/external”; the full terms and abbreviations are introduced later in the text where needed. | Fixed |
| 4 | “这个表质量太低…文字堆砌…项目报告” | Table 1 | Table 1 “Short claim matrix for scope, evidence, and non-claims.” | Table 1 renamed “Claim scope and evidence,” reduced to claim/evidence/boundary columns with concise entries. | Fixed |
| 5 | “每个符号单独成行…” | Table 2 notation table | Several symbols were grouped into one row. | Table 2 now assigns each symbol its own row and clarifies unique meanings for queue, support vector, penalties, losses, service estimates, and replay symbols. | Fixed |
| 7 | “搞成一行…” | Display equation near Eq. 8 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 7 | “搞成一行…” | Display equation near Eq. 8 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 8 | “问题与Table 1一样…杂乱无章” | Proof roadmap table | Proof-roadmap table had dense prose and process-like language. | Proof roadmap rewritten with stage, mathematical role, and term consumed; entries are shorter and aligned with Theorem 1. | Fixed |
| 8 | “搞成一行！” | Display equation near Eq. 14 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 9 | “搞成一行！” | Display equation near Eq. 20 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 9 | “问题同上” | Display equation near Eq. 21 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 10 | “问题同上” | Display equation near Eq. 27 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 10 | “问题同上” | Display equation near Eq. 25 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 10 | “问题同上” | Display equation near Eq. 24 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 10 | “问题同上” | Display equation near Eq. 23 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 10 | “问题同上” | Display equation near Eq. 22 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 11 | “问题同上！” | Display equation near Eq. 31 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 11 | “问题同上” | Display equation near Eq. 30 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 11 | “问题同上” | Display equation near Eq. 29 | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 12 | “这种东西不要出现论文主文当中…太脏” | Lean theorem identifier paragraph | Main text listed long Lean identifiers inline. | Main text now refers to Appendix Table EC.6 for exact Lean crosswalk; long identifiers remain only in the appendix artifact crosswalk. | Fixed |
| 12 | “问题同上” | Implementation paragraph with simulation and algorithm paths | Main text listed internal directories and opt-in hook implementation files. | Implementation rewritten conceptually as theorem-facing policy, legacy launcher, and optional scheduler hints; file-path details removed from the main implementation explanation. | Fixed |
| 12 | “实验过程记录审计残留的表达…系统梳理” | Section “Measured action slices” opening | Main text used audit/process wording before the slice table. | Section rewritten around measured finite slices, robust feasible actions, capacity boundaries, and theorem-facing policy semantics. | Fixed |
| 13 | “这个表不行…残留物质” | Table “Current measured finite slices” | Table used internal bucket names such as `light_control_local` and certificate shorthand. | Table now uses paper-facing bucket names, feasible profiles, boundary profile, replay choice, and legacy cap. | Fixed |
| 13 | “你这是什么公式…看到两个量相减…” | Replay profile-score formula | Formula was a bare score expression. | Formula rewritten as an explicit argmax rule over the measured support envelope; newly introduced symbols were added to the notation table. | Fixed |
| 14 | “有残留；不清晰，脏” | Table “Theorem-policy crosswalk in Scheduleurm” | Table used implementation file names such as `action_model.py`, `service_model.py`, and `skill/scheduler.py`. | Table now maps theorem objects to conceptual Scheduleurm representations: backlog vector, finite configuration action, conservative measured-service row, bounded penalties, oracle-gap reports, and optional scheduler hints. | Fixed |
| 20 | “格式对齐！” | Display equation near theorem-condition certificate | Formula alignment issue. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 22 | No written comment | Highlighted numeric results row | Result table row was highlighted without a text note. | Numeric result retained; surrounding experiment wording was cleaned to state policy-semantics scope and measured-cache comparison. | Fixed |
| 22 | “残留得东西太多了！脏得很” | Markdown path in controlled-production paragraph | Main text cited an internal markdown artifact path. | Live-evidence discussion rewritten without markdown file paths or artifact-run residue. | Fixed |
| 23 | “残留得痕迹太多了！我们是在写论文！” | Production evidence paragraph | Text described trace flags and runner-like closure language. | Section rewritten as controlled and production evidence populations with explicit interpretation and scope. | Fixed |
| 24 | “问题同上，乱七八糟！” | Table “Production and controlled live-evidence ledger” | Table used status/check language and production-run labels. | Table renamed “Controlled and production evidence populations” with population, evidence, and interpretation columns. | Fixed |
| 24 | “这是啥！我的天” | Long SHA-256 hash in Formal artifact section | Main text printed a full checksum hash. | Hash removed from the manuscript body; checksums are now described as part of the supplement manifest. | Fixed |
| 24 | “啥东西这都是…系统梳理” | Reviewer supplement rerun sentence | Main text described a copied reviewer-supplement rerun. | Formal artifact section rewritten as a normal supplement contract: consolidated Lean file, toolchain, build log, crosswalk, manifest, and checksums. | Fixed |
| 25 | “这些东西不应该放到文献综述或引言中么？” | Related Work section | Related-work framing was present but underdeveloped and late. | Related Work now separates queueing/MaxWeight theory from heterogeneous cluster scheduling and adds Decima, DL2, Neely, and Meyn--Tweedie in the correct contexts. | Fixed |
| 27 | “清理干净，全文” | Discussion phrase around `ADMIT_THEOREM_TRACE` and `PROBE_REQUIRED` | Discussion used internal admission/status tokens. | Discussion rewritten without internal tokens; it now states claim scope, production evidence scope, and open directions in prose. | Fixed |
| 28 | “结论写得太潦草…” | Conclusion | Conclusion was too brief and process-like. | Conclusion now recaps the configuration-action model, robust candidate MaxWeight theorem, Lean evidence, measured finite-slice experiments, SOTA-style policy comparison, and scope boundaries. | Fixed |
| 31 | “公式问题同上！” | Appendix display equation | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 31 | “公式问题同上！” | Appendix drift equation | Formula line break. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 32 | “两个公式请给两个编号。有点乱！” | Appendix coordinatewise expectation display | Formula numbering/layout issue. | Not changed; excluded by user instruction on formula-width comments. | Excluded formula layout |
| 33 | “表格问题同上，又乱又有残留！” | Table “Paper theorem to Lean theorem crosswalk” | Lean crosswalk appeared both in the main text and appendix. | Main-text long identifiers removed; appendix crosswalk retained as the formal artifact map because exact theorem names are needed for verification. | Fixed |
| 33 | “表格问题同上，乱糟糟，不清楚！” | Table “Mathematical assumptions and Scheduleurm certificates” | Assumption table used audit-flavored wording. | Table now separates mathematical object, mathematical role, and artifact or empirical certificate in concise rows. | Fixed |
| 34 | “这里更加是破天荒的乱。trash内容！” | Appendix gate-ladder dashboard | Appendix reproduced a large process/status dashboard. | Gate-ladder dashboard removed from the manuscript and replaced by a short Supplementary Evidence Index paragraph. | Fixed |
| 35 | “内容乱糟糟，不清晰还有残留痕迹！” | Appendix status/dashboard continuation | Appendix listed many status rows and non-claim flags. | Dashboard continuation removed; manuscript now directs readers to the supplementary machine-readable evidence package. | Fixed |
| 36 | “这里最后投稿要全部弄好么？” | Code and Data Availability package list | Code availability sounded like an unfinished packaging note. | Section rewritten as a submission artifact package description with source, replay modules, measured service cache, experiment summaries, and Lean supplement. | Fixed |
| 36 | “这里弄干净吧，找个模板看看人家是放哪些东西的…” | Reproduction manifest sentence | Section referenced raw manifest filenames and process scripts. | Section now states what the reproduction script rebuilds and what is intentionally not rerun by default; live launches and external runtime probes are packaged with environment notes. | Fixed |
| 36 | “参考文献核实是都正确的，需要再加一些文献” | Reference list and code availability ending | Reference set lacked some learning-scheduler and queueing-stability context. | Added and verified Decima, DL2, Neely, and Meyn--Tweedie; reference audit file updated from 20 to 24 cited keys. | Fixed |

## Additional Template Cleanup

The INFORMS class inserted a visible template disclaimer beginning “Authors are
encouraged to submit new papers...” on the title page.  It was not a Liang text
annotation, but it matched the user's earlier instruction that template
attention text should not be visibly retained.  The manuscript now overrides
only the title-page abstract block and right header in `paper/main.tex`; the
official class file is not edited.

## 2026-06-17 Follow-Up Supervisor Checks

| Item | Follow-up concern | Before | After | Status |
|---|---|---|---|---|
| Abstract and conclusion Lean wording | “with fixed-family and statewise versions checked in Lean” sounded informal and slightly process-like. | Main text used the phrase “versions checked in Lean.” | Rewritten as “We provide a Lean formalization of the fixed-family and statewise statements” in the abstract and “The fixed-family and statewise statements are formalized in Lean” in the conclusion. | Fixed |
| Table 4 | Measured-slice table was still too close to a numeric experiment log. | Columns were bucket, feasible profiles, boundary, replay choice, and legacy cap. | Rewritten as “Declared measured finite action slices,” with dominant service bottleneck, measured feasible family, excluded boundary, and replay/certificate use. | Fixed |
| Equation 33 | Middle row aligned on the last equals sign rather than the first. | `\epsilon_{\mathrm{est}}=\beta=\alpha_1 &= 0` | Rewritten as `\epsilon_{\mathrm{est}} &= \beta=\alpha_1=0`, aligning the first equals sign with the other rows. | Fixed |
| Tables 7 and 10 profile fonts | Profile entries looked like raw code configuration strings. | Entries such as `gpu=1,cnn=3,llm=10` appeared in table bodies. | Rewritten as paper-facing labels such as “JAX GPU 1; CNN 3; LLM 10.” | Fixed |
| Page 22 host labels | `node007-direct` and `node001` appeared as code-like host tokens. | Corner-case text and table used `\texttt{node007-direct}` and `\texttt{node001}`. | Rewritten as “direct node007 GPU host” in prose and “node001 CPU host” / “node007 GPU host” in the table. | Fixed |
| Related Work placement | Concern that Section 7 might be too late. | Related Work appeared after computational evidence. | Moved Related Work forward after Scope and Contributions and before Model, so the literature frame appears before the formal development. | Fixed |
| Section 7.2 implementation identifier | `adaptive_theorem_maxweight_v1` looked like a code variable in prose. | Section 7.2 named the implementation entry point directly. | Rewritten as “the theorem-facing live hook”; code identifiers are left to the artifact/supplement. | Fixed |
| Appendix Tables 14--17 | Table 14/15 needed to be handled together; Table 16/17 should not disappear or lose gate information. | Table 14 printed long Lean identifiers, Table 15 was separate, and the former Table 16/17 gate ladder had been over-compressed. | Table 14/15 now separate formal proof groups from empirical certificates using paper-facing language; Table 16/17 are expanded evidence-gate summaries that retain finite-domain, LCB/slack, production, live-oracle/global-dispatch, external-policy, runtime-probe, Decima, learning/regime, and future-state safety information. | Fixed |
