# EpiModel prototype v0.1

Research test case: Gollwitzer et al. (2024), [hydrocephalus prediction](https://doi.org/10.1016/j.wneu.2024.09.047), PMID 39276974. The article is CC BY 4.0; `data/methods_input.txt` is study text used as prototype input. The external authors’ repository is [RISCSoftware/shuntclf](https://github.com/RISCSoftware/shuntclf), pinned for the reference runs at commit `1c0ef7020fd2437dc54bc685d8c178b452988ec9`. The external patient CSV and code are not copied here.

## Main experiment: Methods text to environment specification

`prototype.py` has two LLM steps: decompose the Methods text into an analysis plan, then infer dependencies and open questions. A deterministic evaluation compares the output with `data/answer_key.json`, derived from the authors’ released code. `notebooks/epimodel_colab.ipynb` runs this prototype in Google Colab with Ollama. The notebook asks for `epimodel_proto_v0.1.zip`, included in this repository. Upload that ZIP in its first code cell. The ZIP duplicates the prototype source solely to keep the current Colab workflow runnable without a GitHub access token.

The recorded first run is `results/epimodel_R01_llama3.1-8b.json`: llama3.1:8b identified 3 of 5 required packages. This is one run with one model, not a general performance estimate. The notebook in Git has outputs cleared; the JSON preserves the observed result.

## Reference execution of the authors’ code

`notebooks/shuntclf_colab_smoke.ipynb` checks that the authors’ dataset and logistic-regression pipeline run in a cloud environment. `notebooks/shuntclf_logistic_original_cv.ipynb` pins the authors’ commit and evaluates **logistic regression only** with the repository’s default 10-fold × 10-repeat outer and 10-fold inner validation. The first notebook used only 2 outer and 2 inner folds, so its metric values are not comparable with published values.

Results and provenance are in `results/shuntclf_smoke_execution_record.json` and `results/shuntclf_logistic_original_cv_results.json`. The latter has metrics for all 100 outer folds. These runs are reference checks of the authors’ code; they are not the EpiModel agent output or a complete reproduction of every model in the paper.

## Re-running

For the EpiModel prototype, see `notebooks/epimodel_colab.ipynb` or the original `prototype.py` usage below. For the reference runs, open the corresponding notebook in Colab and run cells top to bottom. All package installation in these notebooks occurs in the temporary Colab runtime.

Local prototype requirements (optional): Python 3.10+, Ollama, and `pip install -r requirements.txt`; then run `python prototype.py --mock` for a pipeline check or `python prototype.py --model llama3.1:8b` for the recorded model. Results are written to `results/`.
