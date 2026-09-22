# Qwen3.5-9B large-n_sampled SFT sweeps — runbook

Goal: test whether the SFT negative result (traits decay or stay flat under iterative
self-training) holds at much larger per-cycle sample budgets, on Qwen3.5-9B, in both the
re-init and the continual-learning setting. Only `n_sampled` varies.

| | re-init sweep | continual sweep |
|---|---|---|
| traits | bliss, sycophancy | bliss, sycophancy |
| model | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B |
| n_sampled | 250, 1000, 4000 | 250, 1000, 4000 |
| n_seed | calibrated on 9B (see below) | same value as re-init |
| cycles | 10 | 10 |
| init per cycle | base model | cycle n-1 checkpoint (`--chain-from-prev`) |
| LR | constant 1.5e-4, 5% warmup | constant 1.5e-4, 5% warmup |
| batch size / LoRA rank / epochs | 2 / 16 / 1 | 2 / 16 / 1 |
| seed | 42 | 42 |
| coherence filter | off | off |
| output dir | `outputs/nsampled_sweep_9b_<trait>/seed<n_seed>_nte<n>/` | `outputs/nsampled_sweep_9b_<trait>_continual/seed<n_seed>_nte<n>/` |
| tinker weight tag | `constant-lr-<trait>-9b-nsampled` | `continual-<trait>-9b-nsampled` |

These match the Qwen3-4B `jobs/nsampled_sweep/` runs except for the model, the seed count
(1 here), and the new 4000 value.

## Step 0 — environment

On the cluster the repo venv (`.venv`) already has tinker; keys come from `~/.secrets`.
On any other machine:

```bash
pip install -r requirements.txt && pip install -e .
export TINKER_API_KEY=... OPENAI_API_KEY=... OPENROUTER_API_KEY=...
```

## Step 1 — find the calibration thresholds (needs `cache/` from the 4B sweeps)

The 4B nsampled sweep used n_seed=16 (bliss) and 18 (sycophancy), which were the
calibrated n_seed values at some eval-score threshold on Qwen3-4B. The 9B sweeps
calibrate n_seed to the **same threshold** so cycle-0 persona strength is comparable.

```bash
python jobs/nsampled_sweep_9b/find_4b_thresholds.py
# prints THRESHOLD_BLISS=<T> and THRESHOLD_SYCOPHANCY=<T>
```

`cache/` is gitignored, so this only works on the machine that ran the 4B sweeps. If the
script finds nothing, ask the user which threshold gave n_seed 16 / 18 on Qwen3-4B.
Do not guess: the current `EVAL_THRESHOLDS` in `src/sweep/calibrate.py` may differ from
the thresholds used when the 4B cache was built.

## Step 2 — submit

```bash
cd $HOME/hyperstition
bash jobs/nsampled_sweep_9b/submit_nsampled_sweep_9b.sh <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>
```

Per trait this chains: calibration → (re-init sweep, continual sweep, in parallel) →
eval job after each sweep. 10 SLURM jobs total. `TRAITS="bliss"` submits one trait.

Without SLURM (runs everything sequentially in the foreground; use tmux):

```bash
bash jobs/nsampled_sweep_9b/run_without_slurm.sh <THRESHOLD_BLISS> <THRESHOLD_SYCOPHANCY>
```

## What happens

1. `calibrate_9b_<trait>.sh` runs `src/sweep/calibrate_firstn.py --thresholds T` on 9B:
   binary search over n_seed, training cycle 0 and scoring it, until the eval score
   crosses T. Writes `cache/calibration_<trait>_Qwen_Qwen3.5-9B_constant/calibration_results.json`
   with `threshold_crossings`, `firstn_values` (one value) and `cached_models`
   (n_seed → tinker path of the cycle-0 checkpoint).
2. Both sweep jobs read n_seed = `threshold_crossings[T]` from that cache file and call
   `sweep.py --firstn <n_seed> --use-calibration-cache`. (Do not pass `--thresholds` to
   `sweep.py`: `calibrate()` pads its returned grid to five n_seed values, which would
   turn the 3-run sweep into 15 runs.) Each trait therefore uses its own calibrated
   n_seed; on 9B, bliss calibrated to n_seed=10 at threshold 40.
   - re-init sweep reuses the cached cycle-0 checkpoint for all three n_sampled runs.
   - continual sweep retrains cycle 0 (it needs a saved training *state* to chain from,
     which the calibration checkpoint does not have), then chains cycles 1–6.
3. Cycles 1+ sample `n_sampled` prompts from a disjoint per-cycle slice of
   `src/data/dilemmas.json` (29,280 prompts / 6 cycles ≈ 4,880 per cycle, enough for 4000),
   generate one response each from the previous checkpoint, and train 1 epoch.
4. Weights are logged exactly as before: `save_weights_for_sampler` with the tag in the
   name and a 1-week TTL, path in `cycle<N>/log.txt` (plus `state_log.txt` for the
   continual runs), all paths in `experiment_summary.json`, sweep-level
   `sweep_summary.json` (now also records `model`).
5. Eval jobs run `eval_sweep.py` → `<run>/eval_results.json` and
   `<sweep>/sweep_eval_results.json` (score + coherence per cycle, 10 samples/question).

## Monitoring and resuming

```bash
squeue -u $USER
tail -f logs/cal_<trait>_9b-*.out logs/ns_9b_<trait>*-*.out
ls outputs/nsampled_sweep_9b_<trait>*/seed*_nte*/          # cycle dirs, done.txt per cycle
```

Everything is resumable: re-submitting a sweep skips cycles with `done.txt`; eval skips
runs with `eval_results.json`. Run dirs moved into `<sweep>/extra_n_seed/` are ignored by
eval and plotting (only `seed<N>_nte<M>` dirs at the sweep root are picked up). The nte=4000 run is the long pole (~12,000 steps + 24,000
generations over 6 cycles); if it hits the 24h limit, resubmit the same sweep script
(with `--export=ALL,THRESHOLD=T`) and it continues. Sweep weights expire after 1 week,
so grade promptly.

## Results

```bash
bash jobs/nsampled_sweep_9b/plot_nsampled_sweep_9b.sh
# -> one single-row figure per sweep dir (columns = n_sampled), next to sweep_eval_results.json
python src/analyses/detect_amplification.py --sweep-dir outputs/nsampled_sweep_9b_bliss_continual
```

Compare against the 4B reference: `outputs/nsampled_sweep_bliss_4b_seed16_nte{50,100,500,1000,2000}`
and `..._sycophancy_4b_seed18_nte{60,...,2000}` (six seeds each) and the DPO chain-from-prev
sweeps `outputs/dpo_beta*_10cycles_<trait>_9b_nte{512,1024}_bs4`.
