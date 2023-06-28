# training-code

This repository contains code to perform MLM and text classification fine-tuning.

## Usage

### Install the required dependencies

`requirements.txt` should give you an idea of what you'll need - feel free to `pip install -r requirements.txt` or install things from source depending on which versions you want.

Other packages not listed in `requirements.txt` might also be useful (e.g. `wandb`, `deepspeed` and so on).

### Start training

The training entrypoints are located under [training/](./training) which, as you can probably guess, is based upon [HuggingFace's Trainer class](https://huggingface.co/docs/transformers/main_classes/trainer). As such, all of the command line arguments that can usually be passed in will work here as well.

For convenience's sake, the [scripts/](./scripts) folder contains example starting points you can use:
