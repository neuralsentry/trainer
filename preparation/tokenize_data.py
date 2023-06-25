#!/usr/bin/env python3
import argparse
import logging
import multiprocessing

import datasets
from datasets import load_dataset
from transformers import AddedToken, AutoTokenizer

LOG = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    level=logging.DEBUG,
)


def main() -> None:
    args = _parse_args_from_argv()

    logging.info(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.mask_token is not None:
        LOG.info(f"Set mask token to {args.mask_token}")
        tokenizer.mask_token = args.mask_token

    if args.pad_token is not None:
        LOG.info(f"Set pad token to {args.pad_token}")
        tokenizer.pad_token = args.pad_token

    if args.sep_token is not None:
        LOG.info(f"Set sep token to {args.sep_token}")
        tokenizer.sep_token = args.sep_token

    if args.cls_token is not None:
        LOG.info(f"Set cls token to {args.cls_token}")
        tokenizer.cls_token = args.cls_token

    if args.add_special_tokens is not None:
        # MAINTENANCE(11b): Big fat warning: the snippet below is copy-pasted
        # into ``./training/hf_trainer.py``. Make sure to always keep both
        # implementations in sync.
        special_token_contents = args.add_special_tokens.split(",")
        special_tokens = [
            AddedToken(
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

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    LOG.info(f"Loading dataset from {args.input_file}...")

    raw_dataset: datasets.Dataset
    if args.input_file.endswith(".csv"):
        raw_dataset = load_dataset("csv", data_files=args.input_file, split="train")
    else:
        raw_dataset = load_dataset(args.input_file, split="train")

    LOG.info(f"Tokenizing and trimming dataset...")

    tokenized_dataset: datasets.Dataset
    if args.chunk_sentences:
        tokenized_dataset = raw_dataset.map(
            lambda sample: tokenizer(sample[args.column_name]),
            batched=True,
        )

        tokenized_dataset = tokenized_dataset.map(
            chunk_sentences(args.max_length), batched=True, num_proc=args.n_workers
        )
    else:
        tokenized_dataset = raw_dataset.map(
            lambda sample: tokenizer(
                sample[args.column_name], truncation=True, max_length=args.max_length
            ),
            batched=True,
        )

    if "label" in tokenized_dataset.column_names:
        LOG.info("Renaming `label` column to `labels`...")
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    LOG.info(f"Done! Saving to {args.output_file}...")

    tokenized_dataset.save_to_disk(args.output_file)

    LOG.info("Finished.")


def _parse_args_from_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset tokenizer utility.")
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the input CSV file or HuggingFace dataset.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Path to the output binarized and tokenized file.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer-path",
        required=True,
        help="Path to the HF tokenizer to use.",
    )
    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        required=True,
        help="Max length in tokens before a training example is discarded. Defaults to 2048.",
    )
    parser.add_argument(
        "-s",
        "--add-special-tokens",
        type=str,
        default=None,
        help="Extra special tokens to add to the tokenizer before tokenizing. Comma-separated.",
    )
    parser.add_argument(
        "-c",
        "--column-name",
        type=str,
        default="text",
        help="Name of the column in the input CSV file that contains the text to tokenize. Defaults to 'text'.",
    )
    parser.add_argument(
        "--mask-token",
        type=str,
        default=None,
        help="Token to use for masking. Defaults to the tokenizer's default mask token.",
    )
    parser.add_argument(
        "--pad-token",
        type=str,
        default=None,
        help="Token to use for padding. Defaults to the tokenizer's default pad token.",
    )
    parser.add_argument(
        "--sep-token",
        type=str,
        default=None,
        help="Token to use for separating sentences. Defaults to the tokenizer's default sep token.",
    )
    parser.add_argument(
        "--cls-token",
        type=str,
        default=None,
        help="Token to use for separating sentences. Defaults to the tokenizer's default cls token.",
    )
    parser.add_argument(
        "--chunk-sentences",
        action="store_true",
        default=False,
        help="Whether to chunk sentences for language modeling. Defaults to False.",
    )
    parser.add_argument(
        "-w",
        "--n-workers",
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help="Number of workers to use for CPU-bound tasks. Defaults to half the number of CPU cores.",
    )

    return parser.parse_args()


def chunk_sentences(chunk_size: int):
    def apply(examples: dict):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    return apply


if __name__ == "__main__":
    main()
