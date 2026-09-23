# Agentic tuner — open product questions

Capture design questions that come up while porting the Team TOA hackathon
agent into auto-tune-vllm. Do not treat the hackathon CLI as the target UX.

Status: `open` | `decided` | `deferred`

---

## Q1. Must the user bring an already-running vLLM server?

- **Asked:** 2026-08-19
- **Status:** open (leaning yes to change)
- **Today:** `--vllm-endpoint` + `--oc-pod` are required. Baseline is a
  pre-deployed pod that is never modified.
- **Why it was built that way:** hackathon constraint (pod already existed)
  plus the original bug: restarting vLLM inside the pod kills PID 1.
- **Proposed UX:** user gives **model + optimization target** (and a cluster).
  The controller deploys a baseline, prints perf, *then* experiments.
- **Better?** Yes for a product. Optuna `optimize` already does this
  (`BaselineConfig`, trial 0). The agent should match that shape.
- **Still needed from the user (or a profile):** cluster/kubeconfig, namespace,
  GPU count, image, model PVC. "Model + target" is not enough by itself.
- **Follow-ups:**
  - Pause after baseline for human "continue / stop"?
  - Baseline args = vLLM defaults, or `recipes.vllm.ai` `base_args`?
  - Tear down baseline at the end, or leave the best config running?

---

## Q2. Where do "profiles" live, and are they GPU traces?

- **Asked:** 2026-08-19
- **Status:** decided (naming is confusing; keep both, document)
- **Workload profiles** (ISL/OSL): `auto_tune_vllm/agent/settings.yaml`
- **GPU profiler:** `auto_tune_vllm/agent/profiler/` — capture is **not**
  wired into `create_vllm_pod`

---

## Q3. Should GPU profiling run on every experiment, or on demand?

- **Asked:** 2026-08-19 (implicit: "particularly test profiling")
- **Status:** decided — NOT on every experiment
- **Decision:** overhead is too high; profiling every experiment pod is wasteful.
  `analyze_trace` stays available as an on-demand agent tool. The agent can call it
  if it wants a kernel-level breakdown of a specific config, but it will not run
  automatically. GPU profiling is unrelated to the tuning signal (throughput/latency);
  we don't have a reason to run it all the time.

---

## Q4. Will the agent thoroughly read a vLLM recipe (hardware, TP, specdec, SLOs)?

- **Asked:** 2026-08-19
- **Status:** decided — fetch on controller, inject into first message
- **Decision:** `recipe_fetcher.py` fetches `recipes.vllm.ai/models.json` +
  `{hf_id}.json` at agent startup (on the controller, not the pod). It extracts
  `base_args`, `hardware_overrides[H200]`, `variants`, and `spec_decoding` into
  a structured dict. `format_recipe_briefing()` formats it into a human-readable
  block that is injected directly into the agent's first user message.
  The system prompt's step 0 now says "check your initial context for the
  pre-fetched recipe" instead of curling from inside the pod.
  This avoids the 8KB tool-result truncation that previously cut the recipe.

---

## Q5. If the agent does not run baseline, how does it see a startup profile?

- **Asked:** 2026-08-19
- **Status:** open (leaning: startup Chrome traces not required)
- **Two different "profiles":**
  1. **Startup** (load, `torch.compile`, CUDA graph capture, vLLM warmup).
     Already in pod logs; `fetch_vllm_logs` parses them. Happens when the
     process starts, whether the agent or the controller launched the pod.
  2. **Steady-state forwards** (`Worker.execute_model` after warmup).
     `sitecustomize.py` defaults to range `100-150` — it **skips** startup
     on purpose.
- **If controller deploys baseline and agent never benchmarks it:** logs
  still have startup timings. Chrome traces only appear after enough
  `execute_model` calls (need some load). Profiler must be in the pod at
  **PID 1 start**; cannot attach later.
- **For serving autotune (throughput/TTFT/ITL):** record forward passes,
  not startup traces. Startup traces mix compile/capture and are a poor
  signal for batching knobs.

---

## Q6. How is warmup designed?

- **Asked:** 2026-08-19
- **Decided:** 2026-08-27
- **Status:** decided
- **Scored test is locked** at session start (`settings.yaml` `benchmark:` or CLI
  `--profiles` / `--concurrency` / `--max-seconds`). The agent cannot pick a
  different workload per experiment.
- **Before every scored run:** unscored warmup at a **fixed concurrency**
  for a short duration (`warmup.seconds`, default 15s at conc 8) **or**
  `--warmup-requests N`. Warmup GuideLLM output is discarded. Failure skips
  the scored run.
- **How:** a **separate** GuideLLM process (so warmup concurrency can differ
  from the scored test). GuideLLM *does* have native `--warmup` /
  `--max-requests` (discard the first N seconds or requests of the same run,
  at scored concurrency). We have not switched to that yet.

---

## Q7. Should `--max-iterations` be a hard cap, or a "call done early" budget?

- **Asked:** 2026-09-22 (HERA Gemma 26B live run)
- **Status:** decided (hard cap)
- **Evidence:** `remaining == 5` injected "Call done NOW". Haiku stopped at
  25/30. The loop already writes a report if `done` is never called.
- **Fix:** removed the 5-iteration forced-done nudge. Use all N iterations.

---

## Q8. GuideLLM from the laptop vs in-cluster? GPU profiler?

- **Asked:** 2026-09-22 (same live run)
- **Status:** open
- **Evidence:** scored conc=50 through `oc port-forward` hung (Rich progress
  bar has no newline → timeout never fired; later timeouts misread as OOM).
  Thameem's in-cluster GuideLLM + Prometheus + `--profile` agent are still on
  TOA branches `feat/prom-guidellm-remote` (PR #1 open) and
  `feat/profile-analyser` (PR #2 closed). Not ported. This run did GuideLLM
  only; no Chrome traces.


