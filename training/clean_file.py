import json


def clean_json_file(input_filename, output_filename):
    """
    JSON 파일을 읽어 텍스트 라벨의 공백을 제거하고,
    수정된 내용을 새로운 JSON 파일로 저장합니다.

    Args:
        input_filename (str): 원본 JSON 파일 이름.
        output_filename (str): 저장할 새 JSON 파일 이름.
    """
    # 1. 원본 JSON 파일 읽기
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: '{input_filename}' 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{input_filename}' 파일이 올바른 JSON 형식이 아닙니다.")
        return

    # 2. 데이터 구조를 순회하며 값 수정
    for item in data:
        if 'annotations' in item and isinstance(item['annotations'], list):
            for annotation in item['annotations']:
                if 'result' in annotation and isinstance(annotation['result'], list):
                    for res in annotation['result']:
                        if ('value' in res and isinstance(res['value'], dict) and
                                'text' in res['value'] and isinstance(res['value']['text'], str)):

                            original_text = res['value']['text']

                            # 텍스트의 앞뒤 공백 제거
                            trimmed_text = original_text.strip()

                            if original_text != trimmed_text:
                                # 제거된 공백 수 계산
                                leading_spaces = len(original_text) - len(original_text.lstrip())
                                trailing_spaces = len(original_text) - len(original_text.rstrip())

                                # 값 업데이트
                                res['value']['text'] = trimmed_text
                                if 'start' in res['value']:
                                    res['value']['start'] += leading_spaces
                                if 'end' in res['value']:
                                    res['value']['end'] -= trailing_spaces

    # 3. 수정된 내용을 새로운 파일에 저장
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"성공! 수정된 데이터가 '{output_filename}' 파일로 저장되었습니다.")
    except IOError:
        print(f"오류: '{output_filename}' 파일을 쓰는 데 문제가 발생했습니다.")


# --- 사용 예시 ---
# 원본 파일 이름과 저장할 파일 이름을 지정합니다.
input_file = 'merged_data.json'
output_file = 'cleaned_merged_data.json'

# 함수 실행
clean_json_file(input_file, output_file)