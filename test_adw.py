# test_adw.py

import oracledb
import os

print("--- Starting ADW Connection Test ---")

# config.py의 내용을 그대로 가져옴
ADW_USER = "admin"
ADW_PASSWORD = "oracle_4Uoracle_4U"
ADW_WALLET_PASSWORD = "oracle_4U"
ADW_DSN = "qj5stctdzpuj2amt_high"
WALLET_LOCATION = "/home/opc/ner_project/Wallet_QJ5STCTDZPUJ2AMT"
CLIENT_LIB_DIR = "/usr/lib/oracle/19.19/client64/lib"

# --- 이 코드를 추가하여 TNS_ADMIN 환경 변수를 강제 설정 ---
os.environ['TNS_ADMIN'] = WALLET_LOCATION
print(f"TNS_ADMIN environment variable set to: {WALLET_LOCATION}")
# ---------------------------------------------------------

try:
    print(f"Initializing Oracle Client from: {CLIENT_LIB_DIR}")
    oracledb.init_oracle_client(lib_dir=CLIENT_LIB_DIR)
except Exception as e:
    print(f"[ERROR] During Client Initialization: {e}")

try:
    print("Attempting to connect to ADW...")
    print(f"DSN: {ADW_DSN}")
    
    # 이제 TNS_ADMIN이 설정되었으므로 config_dir 파라미터가 없어도 동작합니다.
    connection = oracledb.connect(
        user=ADW_USER,
        password=ADW_PASSWORD,
        dsn=ADW_DSN,
        wallet_location=WALLET_LOCATION,
        wallet_password=ADW_WALLET_PASSWORD
    )
    print("\n\n✅✅✅ ADW Connection SUCCESS! ✅✅✅\n\n")
    connection.close()

except Exception as e:
    print(f"\n\n❌❌❌ ADW Connection FAILED. Error: {e} ❌❌❌\n\n")
