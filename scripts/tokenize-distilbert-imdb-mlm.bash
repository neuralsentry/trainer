#!/bin/bash

python ./preparation/tokenize_data.py \
--tokenizer-path 'distilbert-base-uncased' \
--input-file 'imdb' \
--output-file './data/imdb-tokenized-mlm' \
--max-length 256 \
--chunk-sentences \
--column-name 'text' \
--n-workers 4
