import json
import git
import os
import re

# --- 설정 ---
REPO_PATH = "."  # 현재 디렉터리의 Git 저장소를 사용
DICT_FILE_PATH = "medical_dict_v7.json"  # Git 히스토리에서 비교할 사전 파일 이름
CHANGE_THRESHOLD = 100  # 재학습을 트리거할 단어 변경 개수
TAG_PREFIX = "model-"  # 모델 버전을 나타내는 태그 접두사
OCI_JOB_OCID = "<YOUR_OCI_DATA_SCIENCE_JOB_OCID>"  # OCI 콘솔에서 복사한 Job OCID로 변경


def count_words_in_dict_from_content(json_content):
    """JSON 문자열 내용에서 총 단어 수를 계산합니다."""
    try:
        data = json.loads(json_content)
        total_words = sum(len(data[category]) for category in data)
        return total_words
    except (json.JSONDecodeError, TypeError):
        return 0


def main():
    print("--- Starting Dictionary Change Check ---")
    repo = git.Repo(REPO_PATH)

    # 1. 'model-' 접두사를 가진 가장 최신 태그를 찾음
    latest_tag = None
    try:
        # 태그를 커밋 날짜순으로 정렬하여 최신 태그를 찾음
        model_tags = [t for t in repo.tags if t.name.startswith(TAG_PREFIX)]
        if not model_tags:
            print("No model tags found. Cannot compare.")
            return
        latest_tag = sorted(model_tags, key=lambda t: t.commit.committed_date)[-1]
        print(f"Latest model tag found: {latest_tag.name}")
    except IndexError:
        print("No model tags found. Cannot compare.")
        return

    # 2. 최신 태그 시점의 사전 파일 내용 가져오기
    try:
        content_prev = repo.git.show(f'{latest_tag.name}:{DICT_FILE_PATH}')
        word_count_prev = count_words_in_dict_from_content(content_prev)
    except git.exc.GitCommandError:
        print(f"Dictionary file not found in tag {latest_tag.name}. Assuming 0 words.")
        word_count_prev = 0

    # 3. 현재 main 브랜치의 사전 파일 내용 가져오기
    try:
        content_current = repo.git.show(f'HEAD:{DICT_FILE_PATH}')
        word_count_current = count_words_in_dict_from_content(content_current)
    except git.exc.GitCommandError:
        print("Dictionary file not found in current HEAD. Assuming 0 words.")
        word_count_current = 0

    print(f"Word count at tag '{latest_tag.name}': {word_count_prev}")
    print(f"Current word count at 'main': {word_count_current}")

    # 4. 변경된 단어 수 확인 및 재학습 트리거
    word_diff = abs(word_count_current - word_count_prev)
    print(f"Word difference since last model deployment: {word_diff}")

    if word_diff >= CHANGE_THRESHOLD:
        print(f"Change threshold of {CHANGE_THRESHOLD} met. Triggering retraining on the local VM...")

        # --- VM 내에서 직접 훈련 스크립트를 실행하는 명령 ---
        # 가상환경의 파이썬 실행 파일 경로
        python_executable = "/home/opc/venv/bin/python"
        # 훈련 스크립트의 전체 경로 (training 폴더 안에 있다고 가정)
        training_script = "/home/opc/ner_project/training/train_step2_finetune.py"

        command = f"{python_executable} {training_script}"

        print(f"Executing command: {command}")
        # os.system을 이용해 셸 명령어 실행
        os.system(command)

        print("\n\nRetraining finished on the VM.")
        print("A new model has been created in the training workspace.")
        print("ACTION REQUIRED: Manually trigger the GitHub Actions pipeline to deploy the new model.")

    else:
        print("No significant changes found. Skipping retraining.")


if __name__ == "__main__":
    main()