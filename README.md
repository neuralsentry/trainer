# accelerate-training-code

A fork of [harubaru/convogpt](https://github.com/harubaru/convogpt).

This is the training code I use to fine-tune the Pygmalion models.

## Warning

I break stuff pretty often in here, and hardcode things with little consideration so I can move quickly. Expect things to not work out of the box. Off the top of my head, you should know that:

- The UFT training code is outdated, and I don't know if it's working.
- The SFT code only works with NeoX-based models + DeepSpeed with at least ZeRO 1 enabled, I believe.

## Quick Start

Assuming you already know the deal about setting up an isolated environment. Install the requirements:

```bash
pip install -r requirements.txt
```

Optionally, if you want to use 8-bit AdamW:

```
pip install bitsandbytes
```

Then start a training run:

```bash
export OMP_NUM_THREADS=4

RUN_NAME="example_run"

BASE_MODEL="EleutherAI/pythia-70m-deduped"
TRAIN_DATASET="./data/sft-small-train.jsonl"
EVAL_DATASET="./data/sft-small-eval.jsonl"
OUTPUT_DIR="./models/"
EPOCHS=2
BATCH_SIZE=1
SAVE_STEPS=50
LEARNING_RATE=1e-5

accelerate launch src/training/sft.py \
    --model "$BASE_MODEL" \
    --train_dataset "$TRAIN_DATASET" \
    --eval_dataset "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --run_name "$RUN_NAME"
```

Things should show up inside `$OUTPUT_DIR` eventually.
