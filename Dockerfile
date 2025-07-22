# Dockerfile

# 1. 베이스 이미지 선택 (Python 3.9 환경)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 시스템 라이브러리 설치 (Oracle Instant Client 의존성)
RUN apt-get update && apt-get install -y libaio1

# 4. requirements.txt 파일을 먼저 복사하여 라이브러리 설치 (캐시 효율성 증대)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트의 나머지 모든 파일들을 복사
COPY . .

# 6. API 서버 실행 (8000번 포트 개방)
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
