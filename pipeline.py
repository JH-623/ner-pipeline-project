# pipeline.py
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from tqdm import tqdm
import logging
import database
import ner_processor
import config
import oracledb

logger = logging.getLogger(__name__)

os.environ['TNS_ADMIN'] = config.WALLET_LOCATION

def run_pipeline(source_table: str, processed_table: str, antibiotic_table: str, failed_table: str):
    """전체 데이터 처리 파이프라인을 실행합니다."""
    logger.info(f"======= PIPELINE START for source table: {source_table} =======")

    source_connection = None  # 변수 초기화
    try:
        # 1. 원본 DB 연결 생성
        source_dsn = f"{config.SOURCE_DB_HOST}:{config.SOURCE_DB_PORT}/{config.SOURCE_DB_SERVICE_NAME}"
        source_connection = oracledb.connect(
            user=config.SOURCE_DB_USER,
            password=config.SOURCE_DB_PASSWORD,
            dsn=source_dsn
        )

        # 2. 데이터 로드 (생성된 연결 객체 전달)
        source_df = database.fetch_source_data(source_table, source_connection)
        if source_df is None or source_df.empty:
            logger.error("No source data found. Terminating pipeline.")
            return

        # 3. 데이터 처리 (이제 연결이 살아있어 LOB 데이터 처리 가능)
        processed_df, antibiotic_df, failed_df = ner_processor.process_dataframe(source_df)

        # 4. ADW에 결과 적재
        logger.info("Connecting to ADW to load data...")
        adw_connection = database.get_adw_connection()
        if adw_connection:
            try:
                database.load_to_adw(processed_df, processed_table.upper(), adw_connection)
                database.load_to_adw(antibiotic_df, antibiotic_table.upper(), adw_connection)
                database.load_to_adw(failed_df, failed_table.upper(), adw_connection)
            finally:
                adw_connection.close()
                logger.info("ADW connection closed.")
        else:
            logger.error("Could not get ADW connection. Results were not saved.")

    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")

    finally:
        # 파이프라인이 끝나면 반드시 원본 DB 연결 종료
        if source_connection:
            source_connection.close()
            logger.info("Source DB connection closed.")

    logger.info(
        f"======= PIPELINE FINISHED for target tables: {processed_table}, {antibiotic_table}, {failed_table} =======")