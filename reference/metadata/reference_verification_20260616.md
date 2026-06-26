# Reference Verification Audit (2026-06-16)

## Scope

This audit checks the contents of `reference/` against external bibliographic
records and the current manuscript bibliography:

- `reference/metadata/gpt_references.tsv`: 30 rows originally collected from
  `md/GPT.md`.
- `reference/papers/` and `reference/pages/`: local PDFs, saved landing pages,
  and access-failure fallbacks.
- `reference/repos/`: cloned repositories associated with scheduler baselines
  and artifacts.
- `paper/references.bib`: 24 references cited by `paper/main.tex`.

## Verification Method

- DOI entries were resolved through Crossref and checked for title, venue, year,
  page/article number, and DOI.
- arXiv entries were checked by arXiv identifier and title.
- USENIX, MLSys, and PMLR entries without DOI fields in the local BibTeX were
  checked against official proceedings pages.
- Local PDFs were checked with `pdfinfo`; sampled PDF text was checked with
  `pdftotext` where the downloaded file name or access route was ambiguous.
- Repository remotes were checked with `git ls-remote <origin> HEAD`.

## Executive Result

- All scholarly items in `reference/metadata/gpt_references.tsv` are real and
  bibliographically resolvable.
- Row 26 is not a paper; it is the INFORMS Operations Research submission
  guidelines page. It should not be used as a literature reference.
- Rows 27-30 are arXiv/preprint duplicates of rows 6, 22, 15, and 5,
  respectively. They are useful as local/open copies but should not be counted
  as independent papers.
- `paper/references.bib` is internally consistent with `paper/main.tex`:
  24 BibTeX keys, 24 cited keys, no missing keys, and no uncited BibTeX entries.
- All 26 local PDF files in `reference/papers/` are valid PDFs according to
  `pdfinfo`.
- `reference/metadata/repo_manifest.tsv` was stale and has been updated to
  include all 13 local repositories.

## Corrections and Caveats

1. `gpt_references.tsv` row 10 is real, but the correct DOI is
   `10.1016/j.jpdc.2025.105138`.
   A wrong nearby JPDC DOI, `10.1016/j.jpdc.2025.105054`, resolves to
   "Efficient GPU-accelerated parallel cross-correlation" and must not be used
   for this scheduler paper.
2. `gpt_references.tsv` row 24 is real, but the correct DOI is
   `10.14778/3685800.3685817`.
   The nearby PVLDB DOI `10.14778/3685800.3685831` resolves to "Lindorm-UWC"
   and must not be used for ResLake.
3. Row 1 is verified as the SC 2023 IADeep paper through DOI
   `10.1145/3581784.3607060`, but the local file currently saved for row 1 is
   the SC23 slide deck, not the camera-ready paper PDF. The landing page is
   saved under `reference/pages/`.
4. Rows 9, 10, 13, and 26 have no local final PDF because the source sites
   blocked or did not expose the PDF through the downloader. This is an access
   limitation, not a bibliographic invalidity.

## GPT Reference List Verification

| id | result | verified bibliographic record | local status |
|---:|---|---|---|
| 1 | verified | IADeep, SC 2023, DOI `10.1145/3581784.3607060` | page saved; slides saved; paper PDF not locally downloaded |
| 2 | verified | arXiv `2405.08754`, "Hierarchical Resource Partitioning on Modern GPUs" | PDF saved |
| 3 | verified | JPDC 2024, DOI `10.1016/j.jpdc.2023.104776` | author copy saved |
| 4 | verified | Gandiva, USENIX OSDI 2018, pages 595-610 | PDF saved |
| 5 | verified | Salus, MLSys 2020; arXiv duplicate is row 30 | PDF saved |
| 6 | verified | Pollux, USENIX OSDI 2021, pages 1-18; arXiv duplicate is row 27 | PDF saved |
| 7 | verified | Decima, SIGCOMM 2019, DOI `10.1145/3341302.3342080`; arXiv `1810.01963` | PDF saved |
| 8 | verified | DL2, arXiv `1909.06040` | PDF saved |
| 9 | verified | RIFLING, Software: Practice and Experience 2022, DOI `10.1002/spe.3066` | publisher access blocked; no local PDF |
| 10 | verified with correction | Topology-aware GPU job scheduling, JPDC 2025, DOI `10.1016/j.jpdc.2025.105138` | publisher access blocked; no local PDF |
| 11 | verified | Stochastic Systems 2022, DOI `10.1287/stsy.2021.0087`; arXiv `1409.0153` | arXiv PDF saved |
| 12 | verified | Probability in the Engineering and Informational Sciences 2006, DOI `10.1017/S0269964806060335` | CWI report copy saved; publisher page saved |
| 13 | verified | Queueing Systems 2006, DOI `10.1007/s11134-006-7586-8` | publisher page saved; no local final PDF |
| 14 | verified | IEEE Transactions on Automatic Control 1992, DOI `10.1109/9.182479` | PDF saved |
| 15 | verified | AISTATS/PMLR 2023, pages 4275-4312; arXiv duplicate is row 29 | PDF saved |
| 16 | verified | ACM TOMPECS 2022, DOI `10.1145/3529375`; arXiv `2011.07401` | PDF saved |
| 17 | verified | Stochastic Systems 2022, DOI `10.1287/stsy.2021.0081`; arXiv `2008.01644` | arXiv PDF saved |
| 18 | verified | PACM Measurement and Analysis of Computing Systems 2020, DOI `10.1145/3379477`; arXiv `1907.10792` | PDF saved |
| 19 | verified | AlloX, EuroSys 2020, DOI `10.1145/3342195.3387547` | PDF saved |
| 20 | verified | Deep learning workload scheduling survey, arXiv `2205.11913` | PDF saved |
| 21 | verified | Themis, USENIX NSDI 2020, pages 289-304 | PDF saved |
| 22 | verified | Gavel, USENIX OSDI 2020, pages 481-498; arXiv duplicate is row 28 | PDF saved |
| 23 | verified | Sia, SOSP 2023, DOI `10.1145/3600006.3613175` | PDF saved |
| 24 | verified with correction | ResLake, PVLDB 17(12):3934-3946, DOI `10.14778/3685800.3685817` | PDF saved |
| 25 | verified | Workload Placement on Heterogeneous CPU-GPU Systems, PVLDB 17(12):4241-4244, DOI `10.14778/3685800.3685845` | PDF saved |
| 26 | real page, not literature | INFORMS Operations Research submission guidelines | not a paper; do not cite as scholarly literature |
| 27 | duplicate/preprint | Pollux arXiv `2008.12260`, duplicate of row 6 | PDF saved |
| 28 | duplicate/preprint | Gavel arXiv `2008.09213`, duplicate of row 22 | PDF saved |
| 29 | duplicate/preprint | MaxWeight discounted UCB arXiv `2209.01126`, duplicate of row 15 | PDF saved |
| 30 | duplicate/preprint | Salus arXiv `1902.04610`, duplicate of row 5 | PDF saved |

## Manuscript Bibliography Verification

The original bibliography audit covered the 20 entries then cited by the
manuscript.  The supervisor revision added four references for learning-based
scheduling and queueing stability context: Decima, DL2, Neely, and
Meyn--Tweedie.  These four entries were checked against the publisher or
official preprint records listed below.

The DOI-backed entries below resolved to matching Crossref records:

- `tassiulas1992stability`
- `stolyar2004maxweight`
- `dai1995stability`
- `sia2023`
- `iadeep2023`
- `le2020allox`
- `carvalho2024workload`
- `gupta2022state`
- `bekker2006admission`
- `altman2006dps`
- `liu2020rlqn`
- `dai2022deep`
- `scully2020simple`
- `deMoura2021lean`
- `mao2019decima`
- `meyn2009markov`
- `neely2010stochastic`

The no-DOI local BibTeX entries below were checked against official proceedings
pages:

- `narayanan2020gavel`: USENIX OSDI 2020, pages 481-498.
- `qiao2021pollux`: USENIX OSDI 2021, pages 1-18.
- `mahajan2020themis`: USENIX NSDI 2020, pages 289-304.
- `xiao2018gandiva`: USENIX OSDI 2018, pages 595-610.
- `yu2020salus`: MLSys 2020 official proceedings page.
- `yang2023maxweight`: PMLR volume 206, pages 4275-4312.
- `peng2019dl2`: arXiv `1909.06040`, "DL2: A Deep Learning-Driven Scheduler for Deep Learning Clusters."

## Supervisor Revision Addendum

The four references added during the Liang-supervisor revision were verified as
follows:

- `mao2019decima`: ACM SIGCOMM 2019, DOI `10.1145/3341302.3342080`.
- `peng2019dl2`: arXiv `1909.06040`, author list and title checked against the
  arXiv record.
- `neely2010stochastic`: Morgan \& Claypool Synthesis Lectures on Communication
  Networks, DOI `10.2200/S00271ED1V01Y201006CNT007`.
- `meyn2009markov`: Cambridge University Press, DOI
  `10.1017/CBO9780511626630`.

## Repository Verification

All 13 local repositories under `reference/repos/` have reachable origin
`HEAD` refs as of this audit. The manifest has been updated with the current
local short commit hashes. These repositories are code artifacts or baseline
implementations, not bibliographic records; they should be referenced in the
paper only when the experiment or artifact discussion actually uses them.
