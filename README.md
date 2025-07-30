미생물 배양검사 결과 분석 자동화 및 MLOps 파이프라인 구축 보고서

1. 프로젝트 개요

1.1. 목표
본 프로젝트의 목표는 비정형 텍스트로 구성된 미생물 배양검사 결과 보고서를 자연어 처리(NLP) 기술을 통해 구조화된 데이터로 자동 변환하는 것이다.
최종적으로는 일회성 변환을 넘어, 지속적인 데이터 변화에 스스로 적응하여 모델을 업데이트하고 배포하는 MLOps(Machine Learning Operations) 파이프라인을 구축하여
운영 가능한 서비스 시스템 수준으로 끌어올리는 것을 목표로 한다.

1.2. 최종 아키텍처
OCI(Oracle Cloud Infrastructure) 클라우드 환경을 기반으로, Git/GitHub Actions를 이용한 CI/CD 파이프라인을 통해 배포가 자동화된 컨테이너 기반 API 서버를 구축했다.
이 시스템은 외부 데이터베이스의 원본 데이터를 처리하여 OCI ADW(Autonomous Data Warehouse)에 적재하며, 도메인 사전의 변경을 감지하여 자동으로 모델을 재학습하고
서비스를 업데이트하는 완전한 MLOps 수명 주기를 갖추었다.

2. 1단계: 초기 파이프라인 구현 및 수동 배포

프로젝트 초기에는 핵심 기능 구현에 집중했다.

인프라 구축: OCI VM(Oracle Linux) 인스턴스를 생성하고 Python 가상환경(venv)을 구성했다.

핵심 로직 개발: transformers 라이브러리 기반의 NER 모델 추론 코드(ner_processor.py)와 oracledb를 이용한 DB 연동 코드(database.py)를 개발했다.

API 서버 구현: FastAPI를 사용하여 외부에서 데이터 처리 요청을 받을 수 있는 REST API(api.py)를 구축했다.

수동 배포 및 실행: 개발된 API 서버를 VM에서 nohup 명령어로 수동 실행했으며, 로컬 PC에서 curl 명령어를 통해 파이프라인을 직접 트리거하는 방식으로 운영했다.

3. 2단계: 문제 해결 및 시스템 안정화 (Troubleshooting)

초기 구현 이후, 시스템을 안정적으로 운영하기 위해 다양한 기술적 문제들을 해결하는 과정을 거쳤다.

문제 현상 (Error)	원인 분석	최종 해결 방안
401/403 인증 오류	OCI API 키, GitHub PAT(Personal Access Token) 인증 방식 문제	OCI fingerprint 수정, GitHub PAT에 repo, write:packages 권한 부여 및 비밀번호 대신 사용
Oracle DB 연결 오류	소스 DB의 암호화(Thick Mode) 요구, ADW의 TNS/Wallet/ACL/내부 경로 문제	Oracle Instant Client 설치 및 Thick 모드 활성화, TNS_ADMIN 환경 변수 설정, sqlnet.ora 경로 수정, ADW 네트워크 ACL에 VCN 및 VM 공용 IP 추가
VM 시스템 오류	디스크 용량 부족(no space left), 셸 스크립트 형식 오류($'\r'), 명령어 경로 부재	OCI 부트 볼륨 확장 및 oci-growfs 실행, dos2unix로 파일 형식 변환, which docker로 절대 경로 확인 후 스크립트에 적용
Git 대용량 파일 오류	단일 파일 2GB 초과로 GitHub Push 실패	Git LFS를 도입하여 대용량 파일을 추적하고, 2GB가 넘는 파일은 zip으로 분할 압축하여 업로드
API 서버 실행 오류	포트 중복 사용(address already in use), 타임아웃	sudo lsof -i :8000으로 기존 프로세스를 찾아 kill -9 <PID>로 기존 프로세스 강제 종료했으며, GitHub Actions의 timeout-minutes 설정을 통해 배포 시간 연장

4. 3단계: MLOps 전환 - CI/CD 자동화

수동 배포의 비효율성을 개선하기 위해 CI/CD 파이프라인을 구축했다.

코드 중앙화: 모든 프로젝트 코드(API, 훈련, 자동화 스크립트)를 GitHub 저장소에서 버전 관리했다.

컨테이너화: API 서버 구동에 필요한 모든 환경을 Dockerfile로 정의하여, 이식성 높고 안정적인 Docker 컨테이너로 패키징했다. 모델 사용을 위해 컨테이너 생성 시 압축된 모델 파일의 압축을 푸는 로직을 포함시켰다.

CI (지속적 통합): OCI 프리티어 제약으로 인해 GitHub Actions를 CI 도구로 채택했다. main 브랜치에 코드가 푸시될 때마다 워크플로우(cicd.yml)가 자동으로 실행되어 Docker 이미지를 빌드하고 GitHub Container Registry(GHCR)에 푸시하도록 구성했다.

CD (지속적 배포): GitHub Actions 워크플로우를 확장하여, CI 성공 후 SSH를 통해 OCI VM에 자동으로 접속하고, 최신 Docker 이미지를 pull 받아 기존 컨테이너를 교체하는 완전 자동화된 배포 파이프라인을 완성했다.

5. 4단계: MLOps 완성 - 운영 및 재학습 자동화

5.1. 운영 자동화: 일일 데이터 처리 파이프라인
수동으로 curl 명령어를 실행하던 방식을 자동화하여 완전한 'hands-off' 운영을 구현했다.

스크립트화: curl 명령어를 날짜 기반의 동적 테이블 이름(예: PROCESSED_DATA_20250716)을 생성하는 로직과 함께 run_pipeline_daily.sh 셸 스크립트로 캡슐화했다.

스케줄링: VM의 Cronjob을 사용하여 이 스크립트를 매일 새벽 3시에 자동으로 실행하도록 등록했다. 이를 통해 매일 새로운 데이터를 처리하고 그날의 결과 테이블을 ADW에 자동으로 생성하는 운영 자동화를 완성했다.

5.2. 자동화된 재학습
데이터 변화에 스스로 적응하는 자가 개선(self-improving) 시스템을 구축했다.

트리거 구현:

의료 사전(medical_dict.json)의 변경량을 감지하는 check_dict_update.py 스크립트를 작성했다. 이 스크립트는 **Git 태그(tag)**를 기준으로 마지막 모델 배포 시점과 현재의 사전 상태를 비교한다.

VM의 Cronjob을 사용하여 이 스크립트를 매일 새벽 2시에 자동으로 실행하도록 스케줄링했다.

자동 재학습 구현:

사전 변경량이 임계치(예: 100건)를 넘으면, 트리거 스크립트가 VM 내에서 직접 훈련 스크립트(train_... .py)를 실행시킨다.

훈련 스크립트는 최신 사전으로 모델을 재학습하고, 성능 평가를 통해 기존 모델보다 성능이 향상되었는지 자동으로 비교한다.

자동 배포 루프 완성:

만약 새 모델의 성능이 더 좋다면, 훈련 스크립트는 자동으로 git push를 실행하여 새 모델과 업데이트된 사전, 성능 기록을 GitHub에 올린다.

이 push는 3단계에서 구축한 CI/CD 파이프라인을 다시 트리거하여, 성능이 검증된 새 모델을 사람의 개입 없이 실제 서비스에 자동 배포함으로써 MLOps의 전체 수명 주기를 완성했다.

6. 결론

본 프로젝트는 단순한 Python 스크립트 실행 환경에서 시작하여, 체계적인 문제 해결 과정을 통해 코드, 데이터, 모델을 버전 관리하고, 빌드/배포/운영/재학습을 자동화하는 완전한 MLOps 파이프라인을 성공적으로 구축했다.
이를 통해 수동 작업의 비효율성을 제거하고, 지속적으로 성능을 개선할 수 있는 안정적이고 확장 가능한 AI 서비스의 기반을 마련했다.
