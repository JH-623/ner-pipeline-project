#!/bin/bash

# API 서버의 주소
API_URL="http://134.185.117.52:8000/run-pipeline/"

# 날짜를 기반으로 동적인 테이블 이름 생성 (예: PROCESSED_DATA_20250716)
TODAY=$(date +%Y%m%d)
SOURCE_TABLE="V_RAW_MICROBE_NEW" # 원본 테이블은 고정이라고 가정
PROCESSED_TABLE="PROCESSED_DATA_${TODAY}"
ANTIBIOTIC_TABLE="ANTIBIOTICS_DATA_${TODAY}"
FAILED_TABLE="FAILED_REPORTS_${TODAY}"

echo "Starting pipeline for date: ${TODAY}"
echo "Target tables: ${PROCESSED_TABLE}, ${ANTIBIOTIC_TABLE}, ${FAILED_TABLE}"

# curl 명령어 실행
curl -X POST "${API_URL}" \
  -H "Content-Type: application/json" \
  -d "{
    \"source_table\": \"${SOURCE_TABLE}\",
    \"processed_table\": \"${PROCESSED_TABLE}\",
    \"antibiotic_table\": \"${ANTIBIOTIC_TABLE}\",
    \"failed_table\": \"${FAILED_TABLE}\"
  }"

echo "\nPipeline trigger request sent."