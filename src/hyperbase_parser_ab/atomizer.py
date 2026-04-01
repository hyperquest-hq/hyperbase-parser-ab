from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


HF_REPO = "hyperquest/atom-classifier"


class Atomizer:
    def __init__(self, model_path:str|None=None):
        model_id = model_path or HF_REPO
        self.model_path = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id)
        self.id2label = self.model.config.id2label

    def atomize(self,
                sentence:str,
                tokens:List[str]|None=None
               ) -> List[Tuple[str, str]]:
        # Tokenize the raw sentence and request offsets
        encoded = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )

        offset_mapping = encoded.pop("offset_mapping")  # remove so model doesn't see it
        word_ids = encoded.word_ids(0)

        with torch.no_grad():
            outputs = self.model(**encoded)

        pred_ids = outputs.logits.argmax(-1)[0].tolist()
        offset_mapping = offset_mapping[0].tolist()

        if tokens is not None:
            # Map provided tokens to model predictions based on character offsets
            return self._map_tokens_to_predictions(sentence, tokens, word_ids, pred_ids, offset_mapping)

        predicted_labels = []
        current_word_id = None
        current_start = None
        current_end = -1
        current_label = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # skip CLS, SEP, etc.

            start, end = offset_mapping[idx]
            label_id = pred_ids[idx]
            label = self.id2label[label_id]

            if word_id != current_word_id:
                # flush previous word
                if current_word_id is not None:
                    word_text = sentence[current_start:current_end]
                    predicted_labels.append((word_text, current_label))

                # start new word
                current_word_id = word_id
                current_start = start
                current_end = end
                current_label = label
            else:
                # same word, extend its span
                current_end = max(current_end, end)

        # flush last word
        if current_word_id is not None:
            word_text = sentence[current_start:current_end]
            predicted_labels.append((word_text, current_label))

        return predicted_labels

    def _map_tokens_to_predictions(self,
                                   sentence: str,
                                   tokens: List[str],
                                   word_ids: List[int|None],
                                   pred_ids: List[int],
                                   offset_mapping: List[List[int]]
                                  ) -> List[Tuple[str, str]]:
        """
        Maps provided tokens to model predictions by finding character offsets
        and assigning the most appropriate label based on overlapping model tokens.
        """
        # Find character positions of each provided token in the sentence
        token_positions = []
        search_start = 0

        for token in tokens:
            pos = sentence.find(token, search_start)
            if pos == -1:
                # Token not found - skip or use fallback
                token_positions.append(None)
            else:
                token_positions.append((pos, pos + len(token)))
                search_start = pos + len(token)

        # For each provided token, collect overlapping model predictions
        result = []
        for token, positions in zip(tokens, token_positions):
            if positions is None:
                # Token not found in sentence - assign default label
                result.append((token, 'C'))
                continue

            token_start, token_end = positions

            # Collect all labels from model tokens that overlap with this token
            overlapping_labels = []
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                model_start, model_end = offset_mapping[idx]

                # Check if model token overlaps with provided token
                if model_start < token_end and model_end > token_start:
                    label = self.id2label[pred_ids[idx]]
                    overlapping_labels.append(label)

            # Assign the most common label, or first label if tie
            if overlapping_labels:
                # Use most common label
                from collections import Counter
                most_common_label = Counter(overlapping_labels).most_common(1)[0][0]
                result.append((token, most_common_label))
            else:
                # No overlap found - use default
                result.append((token, 'C'))

        return result
