#!/usr/bin/env python3
"""Fine-tune Whisper Small on Tamil ASR (Common Voice / FLEURS).

Robust, notebook-free version that fixes the original
`ValueError: No columns in the dataset match the model's forward method signature`
by setting `remove_unused_columns=False`.

Run:
    HF_TOKEN=xxx python scripts/train_whisper_tamil.py
"""
import argparse
import os

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio
from huggingface_hub import login
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
USE_FLEURS = False
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "tamil"
HF_USERNAME = "Venky0411"
HUB_REPO = f"{HF_USERNAME}/whisper-small-ta-saaras"

if USE_FLEURS:
    DATASET, CONFIG, TEXT_COL, HAS_SPEAKER, NEEDS_SPLIT = (
        "google/fleurs",
        "ta_in",
        "transcription",
        False,
        False,
    )
else:
    DATASET, CONFIG, TEXT_COL, HAS_SPEAKER, NEEDS_SPLIT = (
        "abar-uwc/tamil-common-voice_v21",
        None,
        "sentence",
        True,
        True,
    )

MAX_TRAIN_SAMPLES = 8000
MAX_EVAL_SAMPLES = 1000
MAX_STEPS = 2000
BATCH_SIZE = 16
GRAD_ACCUM = 1
LEARNING_RATE = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Whisper for Tamil ASR")
    p.add_argument("--use-fleurs", action="store_true", default=USE_FLEURS)
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--language", default=LANGUAGE)
    p.add_argument("--hub-repo", default=HUB_REPO)
    p.add_argument("--max-train-samples", type=int, default=MAX_TRAIN_SAMPLES)
    p.add_argument("--max-eval-samples", type=int, default=MAX_EVAL_SAMPLES)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--output-dir", default="./whisper-ta")
    p.add_argument("--no-push-to-hub", action="store_true")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    args = parse_args()

    # HF login (token env var / prompt)
    hf_token = args.hf_token
    if not hf_token:
        from getpass import getpass
        hf_token = getpass("HF write token (or set HF_TOKEN env var): ")
    login(token=hf_token)

    # Resolve dataset config
    if args.use_fleurs:
        dataset, config, text_col, has_speaker, needs_split = (
            "google/fleurs",
            "ta_in",
            "transcription",
            False,
            False,
        )
    else:
        dataset, config, text_col, has_speaker, needs_split = (
            DATASET,
            CONFIG,
            TEXT_COL,
            HAS_SPEAKER,
            NEEDS_SPLIT,
        )

    print(f"data={dataset}  config={config}  ->  push to {args.hub_repo}")

    # Load data
    def _load(split):
        return (
            load_dataset(dataset, config, split=split)
            if config
            else load_dataset(dataset, split=split)
        )

    if needs_split:
        full = _load("train")
        parts = full.train_test_split(test_size=0.05, seed=42)
        train, test = parts["train"], parts["test"]
    else:
        train, test = _load("train"), _load("test")

    keep = {"audio", text_col} | ({"client_id"} if has_speaker else set())
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    test = test.remove_columns([c for c in test.column_names if c not in keep])
    print("raw splits:", train, test)

    # Speaker-independent eval for Common Voice
    if has_speaker:
        train_spk = set(train.unique("client_id"))
        test = test.filter(lambda x: x["client_id"] not in train_spk)
        print("speaker-independent test rows:", len(test))

    train = train.shuffle(seed=42)
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples:
        test = test.select(range(min(args.max_eval_samples, len(test))))
    print("using", len(train), "train /", len(test), "test")

    # Processor + feature extraction
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task="transcribe"
    )
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    test = test.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        a = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            a["array"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    train = train.map(prepare, remove_columns=train.column_names, num_proc=2)
    test = test.map(prepare, remove_columns=test.column_names, num_proc=2)

    MAX_LABEL_LENGTH = 448
    train = train.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)
    test = test.filter(lambda x: len(x["labels"]) <= MAX_LABEL_LENGTH)
    print("after length filter:", len(train), "train /", len(test), "test")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor, decoder_start_token_id=model.config.decoder_start_token_id
    )

    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        # Generation predictions may be a tuple (sequences, scores) in some transformers versions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )
        return {
            "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str),
        }

    # Model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=200,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["none"],
        remove_unused_columns=False,  # REQUIRED: dataset only has input_features + labels
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=not args.no_push_to_hub,
        hub_model_id=args.hub_repo,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train,
        eval_dataset=test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # Train
    trainer.train()

    # Final eval
    print("Final eval:", trainer.evaluate())

    # Save + push
    processor.save_pretrained(args.output_dir)
    if not args.no_push_to_hub:
        trainer.push_to_hub(
            dataset=dataset,
            language="ta",
            model_name="Whisper Small Tamil (Saaras-style)",
            finetuned_from=args.model_name,
            tasks="automatic-speech-recognition",
        )
        processor.push_to_hub(args.hub_repo)
        print("Pushed:", args.hub_repo)


if __name__ == "__main__":
    main()
