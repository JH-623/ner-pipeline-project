# -*- coding: utf-8 -*-
import json
import os
import random

# --- 설정 ---
# 팀원들의 라벨링 결과까지 모두 합쳐진 최종 JSON 파일
INPUT_LABELED_FILE = 'cleaned_merged_data.json'
# 우리가 새로 만든 완벽한 사전 파일
DICT_FILE = 'medical_dict_v1.json'
# 최종 증강된 데이터가 저장될 파일
OUTPUT_AUGMENTED_FILE = 'augmented_data_v1.json'


# --------------------

def augment_task_data(task, dictionary, eng_to_kor_dict):
    """하나의 task에 대해 증강을 시도하고, 성공 시 새로운 task를 반환"""
    original_text = task['data']['text']
    annotations = task.get('annotations', [{}])[0].get('result', [])

    swappable_entities = []
    # 사전에 있는 단어들을 찾아서 교체 후보로 등록
    for ann in annotations:
        entity_text = ann.get('value', {}).get('text', '')
        # 한글 -> 영어
        for category, kor_to_eng in dictionary.items():
            if entity_text in kor_to_eng:
                swappable_entities.append((entity_text, kor_to_eng[entity_text], ann))
                break
        # 영어 -> 한글
        if entity_text in eng_to_kor_dict:
            swappable_entities.append((entity_text, eng_to_kor_dict[entity_text], ann))

    if not swappable_entities:
        return None  # 바꿀 단어가 없으면 None 반환

    # 여러 후보 중 하나만 무작위로 선택하여 증강
    original_word, new_word, target_ann = random.choice(swappable_entities)

    # 텍스트 치환
    new_text = original_text.replace(original_word, new_word, 1)

    # 글자 수 차이 계산
    len_diff = len(new_word) - len(original_word)

    # 치환이 일어난 위치
    swap_start_pos = target_ann['value']['start']

    # 새로운 annotation 리스트 생성 및 위치 정보 재계산
    new_annotations_result = []
    for ann in annotations:
        new_ann = json.loads(json.dumps(ann))  # 깊은 복사
        start = new_ann['value']['start']
        end = new_ann['value']['end']

        # 바뀐 단어 자체의 end 위치 조정
        if ann['id'] == target_ann['id']:
            new_ann['value']['text'] = new_word
            new_ann['value']['end'] += len_diff
        # 바뀐 단어 뒤에 있는 다른 라벨들의 start, end 위치 조정
        elif start > swap_start_pos:
            new_ann['value']['start'] += len_diff
            new_ann['value']['end'] += len_diff

        new_annotations_result.append(new_ann)

    # 최종적으로 증강된 task 객체 생성
    augmented_task = json.loads(json.dumps(task))
    augmented_task['data']['text'] = new_text
    augmented_task['annotations'][0]['result'] = new_annotations_result

    return augmented_task


def main():
    print("--- 최종 데이터 증강 시작 (위치 정보 재계산 포함) ---")
    try:
        with open(INPUT_LABELED_FILE, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        with open(DICT_FILE, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)

        # 영어 -> 한글 변환을 위한 역방향 사전 생성
        eng_to_kor_dict = {}
        for category in dictionary:
            for kor, eng in dictionary[category].items():
                eng_to_kor_dict[eng] = kor

        augmented_list = []
        for task in labeled_data:
            # 1. 원본 데이터는 항상 포함
            augmented_list.append(task)
            # 2. 증강된 데이터 생성 시도
            augmented_task = augment_task_data(task, dictionary, eng_to_kor_dict)
            if augmented_task:
                augmented_list.append(augmented_task)

        with open(OUTPUT_AUGMENTED_FILE, 'w', encoding='utf-8') as f:
            json.dump(augmented_list, f, ensure_ascii=False, indent=4)

        print("\n🎉 데이터 증강 완료! 🎉")
        print(f"원본 데이터 {len(labeled_data)}개가 증강되어 총 {len(augmented_list)}개의 데이터가 되었습니다.")
        print(f"결과가 '{OUTPUT_AUGMENTED_FILE}' 파일에 저장되었습니다.")

    except FileNotFoundError as e:
        print(f"[오류] 파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()