# Hyperstition!!

Train a model on a dataset for N cycles and see what happens. Each cycle after the first generates new training data from the previous cycle's model and mixes in a fraction of the original data.

## Setting up environment

You need `tinker`, `tinker_cookbook`, `openai`, and `matplotlib`. Install the pip packages:

```
pip install openai matplotlib
```

You also need tinker. Installation instructions here: https://tinker-docs.thinkingmachines.ai/install

Set your OpenAI and tinker keys:

```
export OPENAI_API_KEY=your-key
export TINKER_API_KEY=your-key
```

## Training

Run from the repo root:

```
python src/train_n_cycles.py --config bliss
```

This trains for 8 cycles on the bliss dataset by default. Outputs go to `outputs/iterative_bliss/`. You can swap in any config:

```
python src/train_n_cycles.py --config misalignment
python src/train_n_cycles.py --config nvidia
python src/train_n_cycles.py --config bad_food
python src/train_n_cycles.py --config sandbag
python src/train_n_cycles.py --config medical
python src/train_n_cycles.py --config financial
```

Other useful flags: `--num-cycles`, `--firstn`, `--batch-size`, `--run-evals`.

## Eval

After training, evaluate the checkpoints:

```
python src/eval.py --config bliss
```

This asks the model eval questions from the config, scores responses with GPT-4o, and saves results + a plot to `outputs/`.

## Configs

Each config in `src/training_configs/` defines:

- `DEFAULT_DATASET` — what to train on
- `EVAL_QUESTIONS` — what to ask during eval
- `SCORE_PROMPT` — how GPT-4o judges responses
- `COHERENCE_PROMPT` — how GPT-4o judges coherence
- `QUERIES_FILE` (optional) — prompts for generating new training data in cycles 1+
