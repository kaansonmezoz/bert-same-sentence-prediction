import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import time

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)


## SSP = Same Sentence Prediction
## Almost identical to Transformers' TextDatasetForNextSentencePrediction
## see: https://github.com/huggingface/transformers/blob/9f72e8f4e1e767c5f608dd135199e592255b8a69/src/transformers/data/datasets/language_modeling.py
class TextDatasetForSameSentencePrediction(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
            ssp_probability=0.5,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.ssp_probability = ssp_probability

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            f"cached_nsp_{tokenizer.__class__.__name__}_{block_size}_{filename}",
        )

        self.tokenizer = tokenizer

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.documents = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.documents[-1]) != 0:
                            self.documents.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.documents[-1].append(tokens)

                logger.info(f"Creating examples from {len(self.documents)} documents.")
                self.examples = []
                for doc_index, document in enumerate(self.documents):
                    self.create_examples_from_document(document, doc_index, block_size)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int):
        """Creates examples for a single document."""

        print("Started creating examples for document => {}/{}".format(doc_index + 1, len(self.documents)))
        doc_start_time = time.time()

        i = 0
        sentences_in_document = len(document)

        ## Every sentence in document will be splitted into two segments.
        ## Every sentence will be tokenized seperately from the other ones.
        while i < sentences_in_document:
            # print("Started processing sentence => {}/{}".format(i+1, sentences_in_document))
            # start_time = time.time()
            sentence = document[i]

            seq_a, seq_b, split_index = self.split_sentence(sentence)
            ##TODO: 5 should be parametrized
            if len(seq_a) < 5:
                is_from_same_sentence = True
                self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))
                ##TODO: Use logger instead of printing ...
                # finish_time = time.time()
                # print("Finished processing {} seconds".format(finish_time - start_time))
                i += 1
                continue

            if random.random() < self.ssp_probability:
                ## Pick a random segment from a sentence in a different document.
                is_from_same_sentence = False
                random_segment = self.pick_random_segment(doc_index, split_index)
                seq_a = random_segment
            else:
                ## Two segments should be from the same sentence
                ## So there is no need to replace one of them.
                is_from_same_sentence = True

            assert len(seq_a) >= 1
            assert len(seq_b) >= 1

            self.examples.append(self.create_example(seq_a, seq_b, is_from_same_sentence))
            i += 1
            # finish_time = time.time()
            # print("Finished processing {} seconds".format(finish_time - start_time))

        doc_finish_time = time.time()
        print("Finished creating examples for document. Time: {} seconds".format(doc_finish_time - doc_start_time))

    def split_sentence(self, sentence: List[int]):
        split_index = len(sentence) // 2
        return sentence[: split_index], sentence[split_index:], split_index

    def pick_random_segment(self, doc_index, token_count):
        for _ in range(10):
            random_sentence = self.pick_random_sentence(doc_index)

            if len(random_sentence) >= token_count:
                return random_sentence[: token_count]

        return random_sentence

    def pick_random_sentence(self, doc_index):
        random_document = self.pick_random_document(doc_index)
        random_sentence_index = self.pick_random_index(len(random_document))
        return random_document[random_sentence_index]

    def pick_random_document(self, doc_index) -> List[int]:
        random_index = self.pick_random_index(len(self.documents), doc_index)
        return self.documents[random_index]

    def pick_random_index(self, len_array, current_index=None):
        for _ in range(10):
            random_index = random.randint(0, len_array - 1)
            if random_index != current_index:
                return random_index

        return random_index

    def create_example(self, seq_a, seq_b, is_from_same_sentence):
        # add special tokens
        input_ids = self.tokenizer.build_inputs_with_special_tokens(seq_a, seq_b)

        # add token type ids, 0 for segment a, 1 for segment b
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(seq_a, seq_b)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "same_sentence_label": torch.tensor(1 if is_from_same_sentence else 0, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
