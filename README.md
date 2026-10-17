# EpiModel prototype v0.1

This repository holds the runnable notebooks, prototype code, and machine-readable run records. The [research document shared with Marios](https://docs.google.com/document/d/1u5uinDQNQsxynVVmOEA6ecQrLoo2AGuQvnyMgsMcD2U/edit?tab=t.xbsw6rwso0ne) records the paper search, screening decisions, Methods-to-code comparison, and interpretation. The two records serve different purposes; the document links back to this repository.

Research test case: Gollwitzer et al. (2024), [hydrocephalus prediction](https://doi.org/10.1016/j.wneu.2024.09.047), PMID 39276974. The article is CC BY 4.0; `data/methods_input.txt` is study text used as prototype input. The external authors’ repository is [RISCSoftware/shuntclf](https://github.com/RISCSoftware/shuntclf), pinned for the reference runs at commit `1c0ef7020fd2437dc54bc685d8c178b452988ec9`. The external patient CSV and code are not copied here.

## Which notebook is which?

These are three separate experiments, not three versions of the same result. The Google Colab files and the Git copies are separate files; editing one does not automatically update the other. The Google Colab column uses the filename verified at each link.

Access note: the three Colab files were owner-only when checked on 17 October 2026. Sharing the research document does not grant access to these linked files; a collaborator needs a separate file permission.

| Purpose | Notebook in this repository | Google Colab | Recorded result |
| --- | --- | --- | --- |
| **Main EpiModel experiment:** infer a computational plan and dependencies from the paper's Methods text | [`notebooks/epimodel_colab.ipynb`](notebooks/epimodel_colab.ipynb) | [`epimodel_colab.ipynb` (EpiModel prototype)](https://colab.research.google.com/drive/1S-5isOKOGy2JrynPYqCS7Ip7YWDvh7QI) | [`results/epimodel_R01_llama3.1-8b.json`](results/epimodel_R01_llama3.1-8b.json) |
| **Initial feasibility check:** run the authors' code with reduced validation | [`notebooks/shuntclf_colab_smoke.ipynb`](notebooks/shuntclf_colab_smoke.ipynb) | [`shuntclf_colab_smoke.ipynb` (initial smoke test)](https://colab.research.google.com/drive/1ylqXbB2C2G8fRi0nsYij3YDUwu8xwLjs) | [`results/shuntclf_smoke_execution_record.json`](results/shuntclf_smoke_execution_record.json) |
| **Closer reference check:** run the authors' logistic-regression code with its original validation settings | [`notebooks/shuntclf_logistic_original_cv.ipynb`](notebooks/shuntclf_logistic_original_cv.ipynb) | [`shuntclf.ipynb` (original-CV run)](https://colab.research.google.com/drive/1Z2IR9LAW9eVXyIgyJ89RaV4pLl_9ZoRo) | [`results/shuntclf_logistic_original_cv_results.json`](results/shuntclf_logistic_original_cv_results.json) |

Start with the **main EpiModel notebook** to inspect the prototype. The other two notebooks establish what can be run from the published study; they are reference executions, not outputs of the EpiModel agents.

## Main experiment: Methods text to environment specification

`prototype.py` has two LLM steps: decompose the Methods text into an analysis plan, then infer dependencies and open questions. A deterministic evaluation compares the output with `data/answer_key.json`, derived from the authors’ released code. `notebooks/epimodel_colab.ipynb` runs this prototype in Google Colab with Ollama. The notebook asks for `epimodel_proto_v0.1.zip`, included in this repository. Upload that ZIP in its first code cell. The ZIP duplicates the prototype source solely to keep the current Colab workflow runnable without a GitHub access token.

The **Decomposition agent** extracts data, preprocessing, models, validation, metrics, software, and unspecified details from the Methods text. The **Dependency agent** uses that plan to propose a Python version and packages, label evidence as stated or inferred, and raise open questions. The **evaluation step is code, not another LLM agent**: it compares the proposal with the answer key. This prototype does not yet create infrastructure or run the study's analysis automatically.

The recorded first run is `results/epimodel_R01_llama3.1-8b.json`: llama3.1:8b identified 3 of 5 required packages. This is one run with one model, not a general performance estimate. The notebook in Git has outputs cleared; the JSON preserves the observed result.

The three identified packages were scikit-learn, numpy, and pandas. It missed imbalanced-learn, although the Methods describes SMOTE, and xgboost, which is present in the released code but not named in the paper. The run also missed several analysis details. The [research document](https://docs.google.com/document/d/1u5uinDQNQsxynVVmOEA6ecQrLoo2AGuQvnyMgsMcD2U/edit?tab=t.xbsw6rwso0ne) explains those observations and their limits.

## Reference execution of the authors’ code

`notebooks/shuntclf_colab_smoke.ipynb` checks that the authors’ dataset and logistic-regression pipeline run in a cloud environment. `notebooks/shuntclf_logistic_original_cv.ipynb` pins the authors’ commit and evaluates **logistic regression only** with the repository’s default 10-fold × 10-repeat outer and 10-fold inner validation. The first notebook used only 2 outer and 2 inner folds, so its metric values are not comparable with published values.

Results and provenance are in `results/shuntclf_smoke_execution_record.json` and `results/shuntclf_logistic_original_cv_results.json`. The latter has metrics for all 100 outer folds. These runs are reference checks of the authors’ code; they are not the EpiModel agent output or a complete reproduction of every model in the paper.

## Re-running

For the EpiModel prototype, see `notebooks/epimodel_colab.ipynb` or the original `prototype.py` usage below. For the reference runs, open the corresponding notebook in Colab and run cells top to bottom. All package installation in these notebooks occurs in the temporary Colab runtime.

Local prototype requirements (optional): Python 3.10+, Ollama, and `pip install -r requirements.txt`; then run `python prototype.py --mock` for a pipeline check or `python prototype.py --model llama3.1:8b` for the recorded model. Results are written to `results/`.
