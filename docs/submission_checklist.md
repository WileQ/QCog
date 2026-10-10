# Submission checklist

Requirement-by-requirement status for the Qollab x IonQ hackathon submission. Every "how to verify" line is a
command or a file path inside this repository.

| # | Requirement | Status | Where / how to verify |
| --- | --- | --- | --- |
| 1 | Uses the Qiskit framework | **Met** | `qiskit.quantum_info.{Statevector, DensityMatrix, Operator, Kraus, partial_trace, schmidt_decomposition}`, `qiskit.circuit.library.{StatePreparation, UnitaryGate}`, `qiskit_aer.AerSimulator`; `qcog/backends/*`, `qcog/circuits/*` |
| 2 | Follows industry best practices on UI accessibility | **Met** | `docs/accessibility.md`; checks in `tests/test_cli_and_report.py::test_model_report_is_accessible`; every figure carries a text description (`qcog.viz`), every chart has a data table, report has skip link / captions / `scope` / focus outlines / reduced-motion-safe styling |
| 3 | Runs on Qollab's free simulators, including IonQ's | **Met** | `qcog run <model> --backend exact|sampled|aer|aer_exact` needs no credentials; optional `--backend ionq` via `qiskit-ionq` (`examples/03_hardware_and_ionq.py`) |
| 4 | Published and runnable on Qollab.xyz under MIT before the deadline | **Action** | publish the repository on Qollab; licence file is `LICENSE` (MIT); see `docs/qollab_runbook.md` |
| 5 | MIT licence for outside repositories it relies on | **Met** | all dependencies are Apache-2.0/BSD/MIT (`ATTRIBUTION.md`); no copyleft dependency |
| 6 | Project page covering what it does, the problem, how to run, how to replicate/extend | **Met** | `docs/project_page.md` (paste-ready), plus `README.md` |
| 7 | Built during the hackathon window; pre-existing material disclosed | **Met** | started from an empty directory; `ATTRIBUTION.md` records the disclosure and the literature usage |
| 8 | Demo video, 2-3 minutes, public link openable without signing in | **Action** | script ready in `docs/demo_video_script.md`; record, upload, paste the link into the project page |
| 9 | Originality: substantially new work, no plagiarism | **Met** | every line written for this project; literature is cited, never copied (`ATTRIBUTION.md`) |

## Non-obvious requirements worth double-checking

* **"Results from IonQ's simulators or hardware neither add nor subtract points."** Nothing in the submission
  depends on IonQ credentials; the default path is local. Verified by running the catalogue with the default
  backends.
* **"Every project must run on a free local simulator."** `qcog evidence` finishes in about a minute on a laptop
  and never touches the network; `qcog report` regenerates every artefact offline.
* **Attribution.** The line *This effort is supported by Qollab & IonQ* appears in the README, the generated
  reports (`report.py` footer), the explorer and the project page - the rules ask for it on the project.
* **Twelve-month availability.** The repository is self-contained (no external data or services); keep the repo
  and the video link alive.
* **Payment method for awards.** Account-level detail, not part of this repository - remember to add it.

## Final pre-submission run

```bash
python -m pytest -q                     # expect: 83 passed
qcog evidence                           # expect: every row PASS
qcog report --out out/qcog_report.html --markdown
python examples/01_first_steps.py
python examples/05_catalogue_report.py
```

Then:

1. publish on Qollab, set the licence to MIT, paste the project page text;
2. record and upload the demo video (script in `docs/demo_video_script.md`), paste the public link;
3. confirm the attribution line is visible on the project page;
4. post at least once on social channels about participating (a winning-team obligation);
5. keep the repository and the video online for twelve months.