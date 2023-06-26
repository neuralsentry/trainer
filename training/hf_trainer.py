import os
import pathlib
import typing as t
from dataclasses import dataclass, field

import torch
import transformers
from datasets import load_from_disk
from profiling import ProfilerCallback, build_profiler_configuration


@dataclass
class ModelArguments:
    tokenizer_path: str = field(metadata={"help": "Path to the HF tokenizer to use."})
    model_path: str = field(metadata={"help": "Path to the HF model to use."})
    max_seq_length: int = field(
        metadata={
            "help": "Max length in tokens before a training example is truncated."
        },
    )
    task_name: str = field(
        metadata={"help": "Name of the task to train on."},
        default="text-classification",
    )
    num_labels: int = field(
        metadata={"help": "Number of labels for the task."}, default=2
    )
    mlm_probability: float = field(
        metadata={"help": "Probability of masking a token."}, default=0.15
    )
    mask_token: t.Optional[str] = field(
        metadata={
            "help": "Token to use for masking. Defaults to the tokenizer's default mask token."
        },
        default=None,
    )
    pad_token: t.Optional[str] = field(
        metadata={
            "help": "Token to use for padding. Defaults to the tokenizer's default pad token."
        },
        default=None,
    )
    sep_token: t.Optional[str] = field(
        metadata={
            "help": "Token to use for separating sentences. Defaults to the tokenizer's default sep token."
        },
        default=None,
    )
    cls_token: t.Optional[str] = field(
        metadata={
            "help": "Token to use for separating sentences. Defaults to the tokenizer's default cls token."
        },
        default=None,
    )


@dataclass
class DataArguments:
    dataset_path: str = field(metadata={"help": "Path to the dataset (arrow)."})
    train_split: float = field(
        metadata={"help": "Fraction of the dataset to use for training."}, default=0.9
    )


@dataclass
class OtherArguments:
    model_load_delay_per_rank: t.Optional[int] = field(
        metadata={
            "help": "Delay loading the model by (this many seconds) * (local_rank)."
        },
        default=None,
    )
    enable_profiler: bool = field(
        metadata={"help": "Whether to profile the training loop."}, default=False
    )
    add_special_tokens: t.Optional[str] = field(
        metadata={
            "help": "Extra special tokens to add to the tokenizer before training. Comma-separated."
        },
        default=None,
    )


def main() -> None:
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            OtherArguments,
            transformers.TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        other_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_path,
    )

    if other_args.model_load_delay_per_rank is not None:
        # When working with constrained system memory, loading the model at the
        # exact same time on all training processes will likely fail due to all
        # the model copies going around. We can delay loading based on
        # local_rank so not all processes are doing this at once, which
        # alleviates the situation. Kinda silly, but it works.
        import time

        time.sleep(other_args.model_load_delay_per_rank * training_args.local_rank)

    device = torch.device("cuda", training_args.local_rank)
    if model_args.task_name == "text-classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_path,
            num_labels=2,
        ).cuda()
    elif model_args.task_name == "mlm":
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_args.model_path,
        ).cuda()
    model.to(device)

    if model_args.mask_token is not None:
        tokenizer.mask_token = model_args.mask_token

    if model_args.pad_token is not None:
        tokenizer.pad_token = model_args.pad_token

    if model_args.sep_token is not None:
        tokenizer.sep_token = model_args.sep_token

    if model_args.cls_token is not None:
        tokenizer.cls_token = model_args.cls_token

    if other_args.add_special_tokens is not None:
        # MAINTENANCE(11b): Big fat warning: the snippet below is copy-pasted
        # into ``./preparation/tokenize_data.py``. Make sure to always keep both
        # implementations in sync.
        special_token_contents = other_args.add_special_tokens.split(",")
        special_tokens = [
            transformers.AddedToken(
                # Heads up: this is very poorly documented in HuggingFace and
                # some old forum discussions mention that it's apparently
                # exclusive to the Rust-based tokenizers? If anything seems
                # funky about the special token behavior, this is a good place
                # to look.
                content,
                lstrip=True,
                rstrip=True,
            )
            for content in special_token_contents
        ]

        _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
            {"additional_special_tokens": special_tokens},
            tokenizer,
            model,
        )

    # Dataset setup.
    tokenized_dataset = load_from_disk(data_args.dataset_path)
    split_dataset = tokenized_dataset.train_test_split(
        train_size=data_args.train_split,
        test_size=1 - data_args.train_split,
        seed=training_args.seed,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=model_args.mlm_probability
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    try:
        # Resume from checkpoint if we have any checkpoints automatically saved
        # by the HF Trainer within the output directory.
        resume_from_checkpoint = (
            len(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))) > 0
        )

        if other_args.enable_profiler:
            profiler_args = build_profiler_configuration()
            with torch.profiler.profile(**profiler_args) as profiler:
                trainer.add_callback(ProfilerCallback(profiler=profiler))
                trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    except KeyboardInterrupt as ex:
        # TODO(11b): Test whether this does what I expect. Idea is to have the
        # trainer save the current state when I interrupt the run so I don't
        # need to keep waiting for a checkpoint step.
        # trainer.save_model()
        # trainer.save_state()
        raise ex

    trainer.save_state()
    trainer.save_model()


def _add_special_tokens_to_tokenizer_and_resize_model_embeddings(
    special_tokens: t.Dict[str, t.Union[str, transformers.AddedToken]],
    tokenizer: transformers.PreTrainedTokenizerBase,
    model: transformers.PreTrainedModel,
):
    tokenizer.add_special_tokens(special_tokens)

    # Size is rounded up to the nearest number divisible by 64 for performance
    # reasons.
    new_size = _nearest_divisible(num=len(tokenizer), divisor=64)
    old_size = model.config.vocab_size

    if new_size == old_size:
        # No resizing needs to be done, let's bail!
        return

    # Need to resize the token embeddings. We initialize the new positions with
    # the mean of the existing ones to cut down on required training time.
    model.resize_token_embeddings(new_size)
    new_positions_count = new_size - old_size

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # This is just to keep the LSP happy.
    assert isinstance(input_embeddings, torch.Tensor)
    assert isinstance(output_embeddings, torch.Tensor)

    input_embeddings_avg = input_embeddings[:-new_positions_count].mean(
        dim=0, keepdim=True
    )
    output_embeddings_avg = output_embeddings[:-new_positions_count].mean(
        dim=0, keepdim=True
    )

    input_embeddings[-new_positions_count:] = input_embeddings_avg
    output_embeddings[-new_positions_count:] = output_embeddings_avg


def _nearest_divisible(num: int, divisor: int) -> int:
    """Returns the nearest number to `num` that is divisible by `divisor`."""
    return (num + divisor - 1) // divisor * divisor


if __name__ == "__main__":
    main()
