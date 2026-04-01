import json

import numpy as np
from numpy.typing import NDArray
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)


def tokenize_and_align_labels(examples: dict[str, list]) -> dict[str, list]:
    """Tokenize each sample and align the original token labels
       to the new subword (tokenized) structure."""

    tokenized_outputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,     # Important for token-based tasks
        return_offsets_mapping=True,  # We'll use this if needed
        padding="max_length",         # or "longest" / "do_not_pad"
        max_length=200                # adjust as needed
    )

    labels_aligned: list[list[int]] = []
    for i, labels in enumerate(examples["labels"]):
        # The tokenizer may split single words into multiple subwords.
        # We create a label list the same length as input_ids,
        # repeating the label for all subwords of the original token.
        word_ids: list[int | None] = tokenized_outputs.word_ids(batch_index=i)
        label_ids: list[int] = []
        previous_word_idx: int | None = None

        for word_idx in word_ids:
            if word_idx is None:
                # This is a special token like [CLS], [SEP], or padding
                label_ids.append(-100)
            else:
                label_ids.append(label_to_id[labels[word_idx]])
            previous_word_idx = word_idx

        labels_aligned.append(label_ids)

    # We don't need offset_mapping during model training, so we remove it
    tokenized_outputs["offset_mapping"] = [None for _ in examples["tokens"]]

    tokenized_outputs["labels"] = labels_aligned
    return tokenized_outputs


def compute_metrics(eval_pred: tuple[NDArray, NDArray]) -> dict[str, float]:
    """Compute accuracy at the token level (simple example).
       You can also compute F1, precision, recall, etc. by ignoring
       the -100 special tokens."""
    logits: NDArray
    labels: NDArray
    logits, labels = eval_pred
    predictions: NDArray = np.argmax(logits, axis=-1)

    # Flatten ignoring -100
    true_predictions: list[int] = []
    true_labels: list[int] = []
    for pred, lab in zip(predictions, labels):
        for p, l in zip(pred, lab):
            if l != -100:  # skip special tokens
                true_predictions.append(p)
                true_labels.append(l)

    results: dict[str, float] = accuracy_metric.compute(
        references=true_labels,
        predictions=true_predictions
    )
    return {"accuracy": results["accuracy"]}


if __name__ == '__main__':
    with open("sentences.jsonl", "rt") as f:
        sentences: list[dict] = [json.loads(line) for line in f]

    dataset_dict: dict[str, list] = {
        "tokens": [sentence["words"] for sentence in sentences],
        "labels": [sentence["types"] for sentence in sentences]
    }

    full_dataset: Dataset = Dataset.from_dict(dataset_dict)

    max_words: int = max([len(sentence["words"]) for sentence in sentences])


    labels: set[str] = set()
    for sentence in sentences:
        labels |= set(sentence["types"])
    print(labels)
    label_to_id: dict[str, int] = {label: i for i, label in enumerate(labels)}
    id_to_label: dict[int, str] = {i: label for label, i in label_to_id.items()}

    dataset = full_dataset.train_test_split(test_size=0.25, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    print("Num train samples:", len(train_dataset))
    print("Num test samples: ", len(test_dataset))


    model_checkpoint: str = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, add_prefix_space=True)

    # Apply to train/test datasets
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    # Remove columns we don't feed directly to the model
    # train_dataset = train_dataset.remove_columns(["tokens", "labels"])
    # test_dataset = test_dataset.remove_columns(["tokens", "labels"])

    # Set format for PyTorch
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id
    )

    accuracy_metric = evaluate.load("accuracy")  # type: ignore[attr-defined]

    training_args: TrainingArguments = TrainingArguments(
        output_dir="./test-roberta-token-classifier",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"  # Set to "tensorboard" if you want logs
    )

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results: dict[str, float] = trainer.evaluate(test_dataset)  # type: ignore[arg-type]
    print("Test set results:", results)

    trainer.save_model("./token-classifier")
