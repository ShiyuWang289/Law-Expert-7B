# Law-Expert-7B — Project Portfolio Brief

## Project Goal
Built an end-to-end legal LLM engineering pipeline to improve Chinese legal QA quality while preserving general-domain capability, and to validate the model through reproducible evaluation, deployment benchmarks, and RAG/agent extensions.

## End-to-End Pipeline
**Data Engineering → SFT → DPO → Eval/Regression → vLLM Deploy/Benchmark → RAG → Lightweight Agent**

- **Data Engineering:** Designed quality-driven sampling and cleaning workflows for legal QA data, then validated data strategy with controlled A/B experiments.
- **SFT (QLoRA):** Fine-tuned Qwen2.5-7B-Instruct for legal response style and structured reasoning.
- **DPO:** Added preference alignment to improve legal grounding and completeness.
- **Eval/Regression:** Standardized automated evaluation and regression alerts (law accuracy, coverage, repetition, hallucination).
- **vLLM Deploy/Benchmark:** Served merged model through OpenAI-compatible API and tuned throughput/latency under concurrency.
- **RAG:** Evaluated retrieval augmentation variants (dense, hybrid, reranker, citation tracing) with ablation and case-level reports.
- **Lightweight Agent:** Implemented minimal function-calling loop with trace-based verification.

## Most Important Quantified Results
- **P3 data-engineering A/B win (random 500 vs engineered 500):**
  - `eval_loss`: **1.1755 → 1.0844** (relative **-7.75%**)
  - auto-eval total: **6.55 → 8.15**
  - head-to-head: **A=2, B=15, Tie=3**
- **General capability stability (CEval):** average remained stable around 79 (**78.83 / 79.42 / 79.05** for Base/SFT/DPO), indicating no catastrophic forgetting.
- **Legal specialization gains:** key-point legal coverage improved **9.9% → 13.3% → 16.7%** (Base/SFT/DPO).
- **vLLM benchmark throughput (FP16):**
  - Concurrency 1/4/8/16: **48.58 / 161.73 / 260.73 / 486.64 TPS**
  - Long output scenario (c=8, max_tokens=256): **263.04 TPS**.
- **RAG findings:** early RAG component added an average **+0.9** score (V1→V2), but case-level variance is high (best **+13.7**, worst **-16.7**) and some advanced stacks reduced quality.
- **Agent status:** function-calling closure is **0/3** traces passing tool-execution + final-answer criteria (known gap).

## Personal Contributions & Skills Demonstrated
- **Data engineering:** candidate pool construction, cleaning-rule rationale, stratified/quality sampling design.
- **Experimentation:** controlled A/B protocol, hyperparameter comparisons, ablation analysis.
- **Evaluation:** metric design and automation, regression-alert thresholds, cross-stage comparison (Base/SFT/DPO).
- **Deployment:** vLLM serving, OpenAI-compatible inference integration, production-lean configuration.
- **Performance tuning:** concurrency sweeps, TTFT/P99/TPS interpretation, capacity-performance tradeoff analysis.
- **RAG analysis:** retrieval strategy comparison, component-level attribution, latency-quality tradeoff diagnosis.

## Limitations (Honest Assessment)
- Some legal coverage metrics rely on strict string matching and may undercount semantically correct paraphrases.
- RAG benefits are not uniformly positive; quality can regress depending on retriever/reranker/prompt stack.
- Lightweight agent function-calling has not yet reached reliable tool-trigger behavior (currently 0/3).

## How to Verify (Existing Scripts)
> Run from repository root unless noted.

```bash
# 1) Reproduce P3 auto-eval report (eval_loss + total score comparison)
python eval/auto_eval_p3.py

# 2) Reproduce standardized eval/regression pipeline (requires model paths in script)
bash eval/run_eval_pipeline.sh

# 3) Reproduce vLLM concurrency benchmark summary (requires running vLLM server)
bash bench/run_concurrency_sweep.sh fp16_baseline
python bench/summarize_sweep.py --tag fp16_baseline

# 4) Reproduce RAG ablation report (requires local service/models)
python rag/phase_d/ablation_experiment.py

# 5) Re-check lightweight agent tool-call pass rate
python agent_minimal/eval_agent_trace.py
```
