# 1. 베이스 이미지 선택 (Python 3.9 환경)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 시스템 라이브러리 설치 (zip 유틸리티 추가)
RUN apt-get update && apt-get install -y libaio1 zip && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt 파일을 먼저 복사하여 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트의 나머지 모든 파일들을 복사
COPY . .

# 6. (추가) 분할 압축된 모델 파일들 압축 해제
#    프로젝트 내의 모든 ..._archive.zip 파일을 찾아 원본 파일로 복원합니다.
RUN find . -name "*_archive.zip" -execdir sh -c 'zip -s 0 "$1" --out "$(basename "$1" _archive.zip).pt"' sh {} \; || echo "No archives to unzip or already unzipped."

# 7. API 서버 실행 (8000번 포트 개방)
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]