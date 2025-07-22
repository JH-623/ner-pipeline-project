# database.py
import oracledb
import pandas as pd
import logging
import config

logger = logging.getLogger(__name__)
logger.warning("<<<<< LOADING NEW DATABASE.PY - THICK MODE ENABLED >>>>>")

# Oracle Client 초기화 (Thick 모드)
try:
    client_lib_dir = "/usr/lib/oracle/19.19/client64/lib"
    oracledb.init_oracle_client(lib_dir=client_lib_dir)
    logger.info(f"Successfully initialized Oracle Client (Thick Mode) from: {client_lib_dir}")
except oracledb.Error as error:
    logger.error(f"Error initializing Oracle Client: {error}")


def fetch_source_data(table_name: str, connection):
    """외부 DB의 지정된 테이블에서 처리할 데이터를 가져옵니다."""
    try:
        logger.info(f"Using provided connection to fetch from table: {table_name}")
        query = f"""
        SELECT
            "내원번호", "환자번호", "성별", "생년월일", "입원일", "검사시행일시",
            "검사명", "검체명_주검체", "검사결과"
        FROM "{table_name}"
        """
        logger.info("Executing query to fetch source data...")
        df = pd.read_sql(query, connection)
        logger.info(f"Fetched {len(df)} rows from the source database.")
        return df
    except Exception as error:
        logger.error(f"Failed to fetch data from source DB: {error}")
        return None


def get_adw_connection():
    """ADW 연결 생성"""
    try:
        connection = oracledb.connect(
            user=config.ADW_USER,
            password=config.ADW_PASSWORD,
            dsn=config.ADW_DSN,
            config_dir=config.WALLET_LOCATION,
            wallet_location=config.WALLET_LOCATION,
            wallet_password=config.ADW_WALLET_PASSWORD
        )
        logger.info("Successfully established connection to ADW.")
        return connection
    except Exception as error:
        logger.error(f"Failed to connect to ADW: {error}")
        return None


def load_to_adw(df, table_name, connection):
    """DataFrame을 ADW에 수동 삽입 방식으로 적재"""
    if df.empty:
        logger.info(f"No data to load into '{table_name}'. Skipping.")
        return

    try:
        table_name = table_name.upper()
        logger.info(f"Loading {len(df)} rows into ADW table: {table_name} (manual insert)")

        cursor = connection.cursor()

        # 기존 테이블 드롭 후 생성
        try:
            cursor.execute(f'DROP TABLE {table_name}')
            logger.info(f"Existing table '{table_name}' dropped.")
        except oracledb.Error as e:
            logger.info(f"Table '{table_name}' did not exist or could not be dropped: {e}")

        # 테이블 생성 (모든 컬럼을 CLOB으로 생성)
        column_defs = ", ".join([f'"{col}" CLOB' for col in df.columns])
        cursor.execute(f'CREATE TABLE {table_name} ({column_defs})')
        logger.info(f"Created table '{table_name}' with CLOB columns.")

        # INSERT
        placeholders = ", ".join([f':{i+1}' for i in range(len(df.columns))])
        insert_query = f'INSERT INTO {table_name} VALUES ({placeholders})'
        data = [tuple(map(str, row)) for row in df.itertuples(index=False)]

        cursor.executemany(insert_query, data)
        connection.commit()
        logger.info(f"Successfully inserted {len(data)} rows into {table_name}.")

    except Exception as e:
        logger.error(f"Failed to load data into table {table_name}: {e}")
