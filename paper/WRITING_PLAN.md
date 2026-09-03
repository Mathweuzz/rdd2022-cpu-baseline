# Paper writing plan

## Start now

Writing should begin before long training runs. The following sections already
have sufficient evidence and can be drafted without speculation:

- motivation and research question;
- RDD2022 description and license;
- exploratory analysis and domain/class distributions;
- leakage risks and split protocol;
- metrics, planned baselines, and reproducibility rules.

These sections should remain living documents. Values derived from manifests
should be generated automatically to avoid disagreement between code and text.

## Writing milestones

| Milestone | Required evidence | Sections to write or freeze |
|---|---|---|
| M0 — now | Audited EDA and protocol | Outline, Introduction v0, Dataset/EDA, Protocol |
| M1 — validated pipeline | One short train/evaluate/export run | Methods v1 and implementation details |
| M2 — primary baseline | Frozen CPU-budgeted run and untouched internal test | Main Results v1 |
| M3 — domain analysis | One shared prediction set sliced across seven domains | Domain analysis and central figures |
| M4 — ablations | Priority experiments complete | Ablations and Discussion |
| M5 — result freeze | Clean rerun and artifact audit | Abstract, Conclusion, submission draft |

All milestones required by the CPU-only scope were completed on 2026-09-03.
The main run, shared domain-sliced prediction file, negative-image analysis,
artifact hashes, English manuscript, and six-page IEEE PDF are frozen. The
higher-resolution, tiling, and repeated-seed studies remain explicitly labeled
as future work rather than missing results.

## Practical rule

Do not wait for every result before writing. Drafting the Dataset and Protocol
sections now reduces rework and exposes which experiments actually support the
paper narrative. The Abstract and Conclusion should be finalized last, after
the results are frozen.
