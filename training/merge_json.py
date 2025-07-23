import json
import os

# 라벨링된 JSON 파일들이 저장된 폴더
data_folder = 'all_labeled_data'
# 합쳐진 파일이 저장될 이름
output_filename = 'merged_data.json'

merged_list = []

# 폴더 안의 모든 json 파일을 순회
for filename in os.listdir(data_folder):
    if filename.endswith('.json'):
        filepath = os.path.join(data_folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # json 파일이 리스트 형태일 경우를 대비
                if isinstance(data, list):
                    merged_list.extend(data)
                else:
                    merged_list.append(data)
            print(f"✅ '{filename}' 파일 로드 완료 (데이터 {len(data)}개)")
        except Exception as e:
            print(f"❌ '{filename}' 파일 처리 중 오류 발생: {e}")

# 합쳐진 데이터를 새 파일로 저장
output_path = os.path.join(data_folder, output_filename)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(merged_list, f, ensure_ascii=False, indent=4)

print("\n" + "="*40)
print(f"🎉 모든 JSON 파일 병합 완료!")
print(f"  - 총 데이터 개수: {len(merged_list)}개")
print(f"  - 저장된 파일: {output_path}")
print("="*40)