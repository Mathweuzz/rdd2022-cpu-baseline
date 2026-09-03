# IEEE conference manuscript

The manuscript is written entirely in English and uses the IEEEtran conference
format.

## Build

From this directory:

```bash
make
```

The generated paper is `main.pdf`. The unmodified upstream IEEEtran package and
the original `bare_conf.tex` skeleton are stored under `ieee-template/vendor/`.
See `ieee-template/SOURCE.md` for provenance and checksums.

## Current status

- The complete English manuscript is compiled from audited dataset, training,
  and frozen internal-test artifacts.
- The current author block identifies Mateus Gomes de Araújo, Computer
  Engineering, University of Brasília (UnB), with contact email
  `mathweuzz@gmail.com`. It can be anonymized later if required by the target
  venue.
- The primary result is a single deterministic CPU-only YOLO11n run. Its
  one-seed design and noncommensurate qualification pilots are stated as
  limitations, not hidden as repeated or comparative experiments.
- Exact artifact hashes and the evaluation command are recorded in
  `RESULTS_PROVENANCE.md`.
- The final venue instructions take precedence over this generic IEEE
  conference setup, particularly page limit, paper size, copyright footer, and
  blind-review requirements.
