#!/bin/bash

# 프로젝트 디렉터리로 이동
cd /home/opc/ner_project/

# 가상환경 활성화
source /home/opc/venv/bin/activate

# 사전 변경 확인 스크립트 실행
python check_dict_update.py
