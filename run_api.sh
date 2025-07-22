#!/bin/bash
echo "Setting TNS_ADMIN and starting NER Pipeline API Server..."

# TNS_ADMIN 환경 변수를 이 스크립트가 실행되는 동안 설정
export TNS_ADMIN=/home/opc/ner_project/Wallet_QJ5STCTDZPUJ2AMT

# uvicorn 서버 실행
uvicorn api:app --host 0.0.0.0 --port 8000
