# -*- coding: utf-8 -*-
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
import os

# --- 사용자 설정 ---
# augment_data.py로 생성한, 증강된 JSON 파일 이름
INPUT_JSON_FILE = 'augmented_data_v1.json'

# 변환된 데이터셋을 저장할 폴더 이름
OUTPUT_DATASET_DIR = 'korean-ner-augmented-v1-dataset'  # 새 데이터셋 폴더


# --------------------

def create_dataset_from_label_studio(tasks):
    """Label Studio 데이터를 Hugging Face Dataset 형식으로 변환하는 함수"""
    processed_examples = []

    for task in tasks:
        # 텍스트는 'data'.'text' 경로 또는 'text' 키에 있을 수 있음
        if 'data' in task and 'text' in task['data']:
            content = task['data']['text']
        elif 'text' in task:  # 다른 형식도 호환
            content = task['text']
        else:
            print(f"Skipping task with unknown content format: {task.get('id')}")
            continue

        # 라벨 정보는 'annotations' 리스트의 첫 번째 항목의 'result'에 있음
        annotations = task.get('annotations', [{}])[0].get('result', [])

        labels = ['O'] * len(content)

        for ann in annotations:
            value = ann.get('value')
            if not value or 'labels' not in value:
                continue

            tag = value['labels'][0]
            start = value['start']
            end = value['end']

            if start < end:
                labels[start] = f'B-{tag}'
                for i in range(start + 1, end):
                    labels[i] = f'I-{tag}'

        tokens = []
        token_labels = []
        current_word = ""
        for char, label in zip(content, labels):
            if char.isspace():
                if current_word:
                    tokens.append(current_word)
                    token_labels.append(current_word_label)
                    current_word = ""
            else:
                if not current_word:
                    current_word_label = label
                current_word += char

        if current_word:
            tokens.append(current_word)
            token_labels.append(current_word_label)

        processed_examples.append({'tokens': tokens, 'ner_tags': token_labels})

    return processed_examples


def main():
    print(f"'{INPUT_JSON_FILE}' 파일을 읽어옵니다...")
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            # Label Studio Export 파일은 리스트이므로, 바로 로드합니다.
            # ['examples'] 부분을 제거한 것이 핵심 수정 사항입니다.
            data = json.load(f)
    except FileNotFoundError:
        print(f"[오류] '{INPUT_JSON_FILE}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 확인해주세요.")
        return
    except Exception as e:
        print(f"JSON 파일을 읽는 중 오류 발생: {e}")
        return

    print("데이터를 학습 가능한 형태로 변환합니다...")
    processed_data = create_dataset_from_label_studio(data)

    all_tags = sorted(list(set(tag for item in processed_data for tag in item['ner_tags'])))

    # 'O' 태그가 항상 포함되도록 보장
    if 'O' not in all_tags:
        all_tags.append('O')

    label_map = {label: i for i, label in enumerate(all_tags)}

    for item in processed_data:
        item['ner_tags'] = [label_map.get(tag, label_map['O']) for tag in item['ner_tags']]

    print("\n생성된 라벨 목록:")
    print(all_tags)

    train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

    print(f"\n훈련용 데이터: {len(train_data)}개, 검증용 데이터: {len(val_data)}개로 분리합니다.")

    features = Features({
        'tokens': Sequence(Value('string')),
        'ner_tags': Sequence(ClassLabel(names=all_tags))
    })

    train_dataset = Dataset.from_list(train_data, features=features)
    val_dataset = Dataset.from_list(val_data, features=features)

    final_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    print(f"\n변환된 데이터셋을 '{OUTPUT_DATASET_DIR}' 폴더에 저장합니다...")
    if not os.path.exists(OUTPUT_DATASET_DIR):
        os.makedirs(OUTPUT_DATASET_DIR)
    final_dataset.save_to_disk(OUTPUT_DATASET_DIR)

    print("\n🎉 데이터 준비 완료! 🎉")
    print("이제 이 데이터셋으로 최종 모델 훈련을 진행할 수 있습니다.")


if __name__ == "__main__":
    main()