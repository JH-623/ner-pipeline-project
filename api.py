# api.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging
import os
from pipeline import run_pipeline

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NER Processing Pipeline API",
    description="데이터를 받아 NER 파이프라인을 비동기적으로 실행하는 API"
)


# API 요청 Body 모델
class PipelineRequest(BaseModel):
    source_table: str
    processed_table: str
    antibiotic_table: str
    failed_table: str


@app.post("/run-pipeline/", status_code=202)
async def trigger_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """NER 처리 파이프라인 실행을 요청합니다."""
    logger.info(f"Received pipeline request: {request.dict()}")

    # 백그라운드에서 파이프라인 함수 실행
    background_tasks.add_task(
        run_pipeline,
        request.source_table,
        request.processed_table,
        request.antibiotic_table,
        request.failed_table
    )

    return {"message": "Pipeline execution started in the background."}


@app.get("/")
def read_root():
    return {"status": "NER Pipeline API is running."}