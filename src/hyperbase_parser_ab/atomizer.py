from collections import Counter

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

HF_REPO: str = "hyperquest/atom-classifier"

WordPrediction = tuple[str, str]
WordPredictionTopK = tuple[str, list[tuple[str, float]]]


class Atomizer:
    def __init__(self, model_path: str | None = None) -> None:
        model_id: str = model_path or HF_REPO
        self.model_path: str = model_id
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_id, use_fast=True
        )
        self.model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
            model_id
        )
        assert self.model.config.id2label
        self.id2label: dict[int, str] = self.model.config.id2label

    def atomize(
        self,
        sentence: str,
        tokens: list[str] | None = None,
        top_k: int = 1,
    ) -> list[WordPrediction] | list[WordPredictionTopK]:
        # Tokenize the raw sentence and request offsets
        encoded = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, return_offsets_mapping=True
        )

        offset_mapping = encoded.pop("offset_mapping")  # remove so model doesn't see it
        word_ids: list[int | None] = encoded.word_ids(0)

        with torch.no_grad():
            outputs = self.model(**encoded)

        probs: torch.Tensor = torch.softmax(outputs.logits[0], dim=-1)
        pred_ids: list[int] = probs.argmax(-1).tolist()
        offset_mapping = offset_mapping[0].tolist()

        if tokens is not None:
            # Map provided tokens to model predictions based on character offsets
            return self._map_tokens_to_predictions(
                sentence, tokens, word_ids, pred_ids, probs, offset_mapping, top_k
            )

        results: list = []
        current_word_id: int | None = None
        current_start: int | None = None
        current_end: int = -1
        current_first_idx: int | None = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  # skip CLS, SEP, etc.

            start: int
            end: int
            start, end = offset_mapping[idx]

            if word_id != current_word_id:
                # flush previous word
                if current_first_idx is not None:
                    word_text: str = sentence[current_start:current_end]
                    results.append(
                        self._format_prediction(
                            word_text, current_first_idx, pred_ids, probs, top_k
                        )
                    )

                # start new word
                current_word_id = word_id
                current_start = start
                current_end = end
                current_first_idx = idx
            else:
                # same word, extend its span
                current_end = max(current_end, end)

        # flush last word
        if current_first_idx is not None:
            word_text = sentence[current_start:current_end]
            results.append(
                self._format_prediction(
                    word_text, current_first_idx, pred_ids, probs, top_k
                )
            )

        return results

    def _format_prediction(
        self,
        text: str,
        idx: int,
        pred_ids: list[int],
        probs: torch.Tensor,
        top_k: int,
    ) -> WordPrediction | WordPredictionTopK:
        if top_k <= 1:
            return (text, self.id2label[pred_ids[idx]])

        k: int = min(top_k, len(self.id2label))
        top_probs, top_indices = torch.topk(probs[idx], k)
        return (
            text,
            [
                (self.id2label[label_id], prob)
                for label_id, prob in zip(
                    top_indices.tolist(), top_probs.tolist(), strict=True
                )
            ],
        )

    def _map_tokens_to_predictions(
        self,
        sentence: str,
        tokens: list[str],
        word_ids: list[int | None],
        pred_ids: list[int],
        probs: torch.Tensor,
        offset_mapping: list[list[int]],
        top_k: int,
    ) -> list[WordPrediction] | list[WordPredictionTopK]:
        """
        Maps provided tokens to model predictions by finding character offsets
        and assigning the most appropriate label based on overlapping model tokens.
        For top_k > 1, returns the first overlapping subword's full distribution.
        """
        # Find character positions of each provided token in the sentence
        token_positions: list[tuple[int, int] | None] = []
        search_start: int = 0

        for token in tokens:
            pos: int = sentence.find(token, search_start)
            if pos == -1:
                # Token not found - skip or use fallback
                token_positions.append(None)
            else:
                token_positions.append((pos, pos + len(token)))
                search_start = pos + len(token)

        # For each provided token, collect overlapping model predictions
        result: list = []
        for token, positions in zip(tokens, token_positions, strict=True):
            if positions is None:
                # Token not found in sentence - assign default label
                result.append(self._default_prediction(token, top_k))
                continue

            token_start: int
            token_end: int
            token_start, token_end = positions

            # Collect all labels from model tokens that overlap with this token
            first_overlap_idx: int | None = None
            overlapping_labels: list[str] = []
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                model_start: int
                model_end: int
                model_start, model_end = offset_mapping[idx]

                # Check if model token overlaps with provided token
                if model_start < token_end and model_end > token_start:
                    if first_overlap_idx is None:
                        first_overlap_idx = idx
                    overlapping_labels.append(self.id2label[pred_ids[idx]])

            if not overlapping_labels:
                # No overlap found - use default
                result.append(self._default_prediction(token, top_k))
                continue

            if top_k <= 1:
                # Use most common label, or first label if tie
                most_common_label: str = Counter(overlapping_labels).most_common(1)[0][
                    0
                ]
                result.append((token, most_common_label))
            else:
                assert first_overlap_idx is not None
                result.append(
                    self._format_prediction(
                        token, first_overlap_idx, pred_ids, probs, top_k
                    )
                )

        return result

    def _default_prediction(
        self, token: str, top_k: int
    ) -> WordPrediction | WordPredictionTopK:
        if top_k <= 1:
            return (token, "C")
        return (token, [("C", 1.0)])
