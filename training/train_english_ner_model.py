# -*- coding: utf-8 -*-
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

# ------------------- 사용자 설정 ------------------- #
DATASET_DIR = 'medical-ner-korean-ready'
BASE_MODEL_NAME = 'xlm-roberta-base'
OUTPUT_MODEL_DIR = 'english-ner-model'


# ---------------------------------------------------- #

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = {}
    all_metrics["f1"] = f1_score(true_labels, true_predictions)
    all_metrics["precision"] = precision_score(true_labels, true_predictions)
    all_metrics["recall"] = recall_score(true_labels, true_predictions)
    return all_metrics


# --- 스크립트 메인 실행부 ---
if __name__ == "__main__":
    print("--- 2단계: 영어 기반 모델 훈련 시작 ---")

    raw_datasets = load_from_disk(DATASET_DIR)
    label_names = raw_datasets['train'].features['ner_tags'].feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}
    num_labels = len(label_names)

    print("\n라벨 목록 확인:")
    print(label_names)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n훈련 장치: {device.upper()}")

    # 최신 라이브러리 환경에 맞는 TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",  # 매 에폭마다 평가
        save_strategy="epoch",  # 매 에폭마다 저장
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=50,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- 모델 훈련을 시작합니다. ---")
    trainer.train()

    print("\n훈련이 완료되었습니다. 최종 모델을 저장합니다.")
    trainer.save_model(OUTPUT_MODEL_DIR)

    print(f"\n🎉 2단계 완료! 🎉")
    print(f"1차 학습된 모델이 '{OUTPUT_MODEL_DIR}' 폴더에 저장되었습니다.")