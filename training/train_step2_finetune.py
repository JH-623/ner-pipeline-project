# -*- coding: utf-8 -*-
import os
import json
import shutil
import torch
import numpy as np
import optuna
import re
import glob
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification,
    AddedToken
)
from seqeval.metrics import f1_score, precision_score, recall_score


# ------------------- 버전 관리 함수 ------------------- #
def get_latest_versions(base_path):
    """
    지정된 경로에서 가장 최신 버전의 사전과 모델 경로를 찾아내고,
    다음 버전 번호를 계산하여 새로운 경로를 반환합니다.
    """
    # 사전 파일 찾기 (예: medical_dict_v7.json)
    dict_pattern = os.path.join(base_path, 'medical_dict_v*.json')
    dict_files = sorted(glob.glob(dict_pattern))
    if not dict_files:
        raise FileNotFoundError(f"사전 파일을 찾을 수 없습니다: {dict_pattern}")

    latest_dict_file = dict_files[-1]
    match = re.search(r'_v(\d+)\.json$', latest_dict_file)
    current_version = int(match.group(1)) if match else 0

    next_version = current_version + 1

    # 새 모델 저장 경로 생성
    output_model_dir = os.path.join(base_path, f'final-korean-ner-model-optimized_v{next_version}')

    print(f"🔄 자동 버전 감지:")
    print(f"   - 최신 사전 파일: {os.path.basename(latest_dict_file)} (버전 {current_version})")
    print(f"   - 새 모델 저장 경로: {os.path.basename(output_model_dir)} (버전 {next_version})")

    return latest_dict_file, output_model_dir


# ------------------- 사용자 설정 ------------------- #
BASE_DIR = '/home/opc/ner_project/training_workspace'  # OCI VM 내의 실제 작업 경로로 수정 필요
DATASET_DIR = os.path.join(BASE_DIR, 'korean-ner-augmented-v1-dataset')
BASE_MODEL_PATH = os.path.join(BASE_DIR, 'english-ner-model')

# 아래 두 줄이 자동으로 설정되도록 변경되었습니다.
MEDICAL_DICT_PATH, OUTPUT_MODEL_DIR = get_latest_versions(BASE_DIR)

# ------------------- 전역 변수 설정 ------------------- #
tokenizer = None
tokenized_datasets = None
label_names = None
id2label = None
label2id = None
num_labels = None


# ------------------- 도메인 단어/레이블 처리 함수 ------------------- #
def load_domain_tokens(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        med_dict = json.load(f)
    token_set = set()
    print(f"🔍 사전에서 '{', '.join(med_dict.keys())}' 카테고리를 발견했습니다.")
    for category, term_dict in med_dict.items():
        for kor, eng in term_dict.items():
            token_set.add(kor.strip())
            token_set.add(eng.strip())
    return list(token_set)


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
            # 서브워드에는 -100을 할당하여 손실 계산에서 제외
            new_labels.append(-100)
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
    return {
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
    }


# ------------------- Optuna 최적화 관련 함수 ------------------- #
def model_init():
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_PATH,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def objective(trial):
    # 임시 출력 디렉터리 설정 (메모리 관리)
    temp_output_dir = f"./tmp_trial_{trial.number}"

    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        num_train_epochs=trial.suggest_int("num_train_epochs", 5, 10),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        disable_tqdm=True,
        report_to=[],
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"[Trial {trial.number}] Eval result: {eval_result}")

    # 평가 후 임시 디렉터리 정리
    shutil.rmtree(temp_output_dir, ignore_errors=True)

    return eval_result.get("eval_f1", 0.0)


# ------------------- 훈련 실행부 ------------------- #
if __name__ == "__main__":
    print("🚀 1. 데이터 및 토크나이저 준비...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    custom_tokens = load_domain_tokens(MEDICAL_DICT_PATH)
    added_tokens = [AddedToken(t, single_word=True) for t in custom_tokens]
    tokenizer.add_tokens(added_tokens)
    print(f"✅ 추가된 도메인 토큰 수: {len(added_tokens)}")

    raw_datasets = load_from_disk(DATASET_DIR)
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    label_names = raw_datasets['train'].features['ner_tags'].feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in id2label.items()}
    num_labels = len(label_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 훈련 장치: {device.upper()}")

    print("\n🚀 2. Optuna 하이퍼파라미터 탐색 시작 (5 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print(f"\n🎉 탐색 완료! 최적 F1 점수: {study.best_value:.4f}")
    print("📈 최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"   - {key}: {value}")

    print("\n🚀 3. 최적 하이퍼파라미터로 최종 모델 훈련 시작...")

    best_params = study.best_params

    final_training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir="./logs",
        report_to="none",
        seed=42
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    final_trainer = Trainer(
        model=model_init(),
        args=final_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    final_trainer.train()

    print("\n💾 최종 모델 저장 중...")
    final_trainer.save_model(OUTPUT_MODEL_DIR)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"\n🎉 최적화된 한국어 NER 모델 저장 완료: '{OUTPUT_MODEL_DIR}'")