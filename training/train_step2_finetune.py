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

    return latest_dict_file, output_model_dir, next_version


# ------------------- 압축/해제 헬퍼 함수 ------------------- #
def unzip_model_if_needed(model_path):
    """지정된 경로의 모델 파일 압축을 해제합니다."""
    print(f"Checking for zipped models in {model_path}...")
    # optimizer.pt 같은 대용량 파일이 압축되어 있는지 확인
    zip_files = glob.glob(os.path.join(model_path, '**', '*_archive.zip'), recursive=True)
    if not zip_files:
        print("No zipped models found to unzip.")
        return

    for zip_file in zip_files:
        # 원본 파일 이름은 '_archive'를 제외한 이름으로 가정 (예: optimizer.pt)
        original_filename = os.path.basename(zip_file).replace('_archive.zip', '.pt')  # .pt 외 다른 확장자도 고려 필요 시 수정
        if 'optimizer' not in original_filename:  # optimizer.pt 외 다른 파일일 경우를 대비
            original_filename = os.path.basename(zip_file).replace('_archive.zip', '.safetensors')

        output_path = os.path.dirname(zip_file)
        original_filepath = os.path.join(output_path, original_filename)

        # 이미 원본 파일이 있으면 건너뛰기
        if os.path.exists(original_filepath):
            print(f"{original_filename} already exists. Skipping unzip.")
            continue

        print(f"Unzipping {zip_file} to {original_filepath}...")
        command = f'zip -s 0 "{zip_file}" --out "{original_filepath}"'
        result = os.system(command)
        if result == 0:
            print("Unzip successful.")
        else:
            raise RuntimeError(f"Error unzipping file: {zip_file}")


def zip_and_cleanup_large_files(model_path):
    """지정된 경로에서 2GB가 넘는 파일을 찾아 분할 압축하고 원본을 삭제합니다."""
    print(f"Checking for large files to zip in {model_path}...")
    large_files = []
    # 2GB = 2 * 1024 * 1024 * 1024 bytes
    size_limit_bytes = 2 * 1024 * 1024 * 1024
    for dirpath, _, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath) and not os.path.islink(filepath):
                if os.path.getsize(filepath) > size_limit_bytes:
                    large_files.append(filepath)

    if not large_files:
        print("No large files (>2GB) found to zip.")
        return

    for large_file in large_files:
        print(f"Found large file: {large_file}")
        archive_name = os.path.splitext(large_file)[0] + '_archive.zip'
        print(f"Zipping to {archive_name}...")
        command = f'zip -s 1g "{archive_name}" "{large_file}"'
        result = os.system(command)
        if result == 0:
            print(f"Zip successful. Deleting original file: {large_file}")
            os.remove(large_file)
        else:
            raise RuntimeError(f"Error zipping file: {large_file}")


# ------------------- 사용자 설정 ------------------- #
BASE_DIR = '/home/opc/ner_project/training'  # OCI VM 내의 실제 작업 경로
DATASET_DIR = os.path.join(BASE_DIR, 'korean-ner-augmented-v1-dataset')
BASE_MODEL_PATH = os.path.join(BASE_DIR, 'english-ner-model')
MEDICAL_DICT_PATH, OUTPUT_MODEL_DIR, next_version = get_latest_versions(BASE_DIR)

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
    # --- 학습 전 베이스 모델 압축 해제 ---
    print("\n🚀 0. 베이스 모델 압축 해제 확인...")
    unzip_model_if_needed(BASE_MODEL_PATH)

    print("\n🚀 1. 데이터 및 토크나이저 준비...")
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

    # --- 학습 후 새 모델 압축 및 정리 ---
    print("\n🚀 4. 새로 생성된 대용량 모델 파일 압축 및 정리...")
    zip_and_cleanup_large_files(OUTPUT_MODEL_DIR)

    print("\n🚀 5. 새 모델 성능 평가 및 자동 배포 시작...")

    # 5-1. 방금 훈련한 새 모델의 F1 점수 가져오기
    eval_results = final_trainer.evaluate()
    new_f1_score = eval_results.get("eval_f1", 0.0)
    print(f" - 새 모델 F1 점수: {new_f1_score:.4f}")

    # 5-2. 현재 서비스 중인 모델의 성능 지표 로드
    production_metrics_file = os.path.join(BASE_DIR, 'production_metrics.json')
    try:
        with open(production_metrics_file, 'r') as f:
            prod_metrics = json.load(f)
        prod_f1_score = prod_metrics.get("f1", 0.0)
    except FileNotFoundError:
        prod_f1_score = 0.0  # 파일이 없으면 0점으로 간주하여 항상 업데이트

    print(f" - 현재 모델 F1 점수: {prod_f1_score:.4f}")

    # 5-3. 성능 비교 후 자동 배포 결정
    if new_f1_score > prod_f1_score:
        print("\n✅ 성능 향상! 새 모델을 Git에 푸시하여 자동 배포를 시작합니다.")

        # 새 성능 지표를 파일에 저장 (다음 비교를 위해)
        with open(production_metrics_file, 'w') as f:
            json.dump({"f1": new_f1_score}, f)

        # Git 명령어 실행
        try:
            os.system('git config --global user.name "AutoTrain Bot"')
            os.system('git config --global user.email "bot@example.com"')
            os.system('git add .')
            commit_message = f"Auto-train: Update model to v{next_version} with F1 score {new_f1_score:.4f}"
            os.system(f'git commit -m "{commit_message}"')
            os.system('git push origin main')
            print("✅ Git push 완료! CD 파이프라인이 새 모델을 배포합니다.")
        except Exception as e:
            print(f"❌ Git push 중 오류 발생: {e}")

    else:
        print("\n❌ 성능이 향상되지 않았으므로 현재 모델을 유지합니다.")