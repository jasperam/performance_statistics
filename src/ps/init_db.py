import pandas as pd
import os
import datetime

from jt.utils.db import PgSQLLoader
from jt.utils.fs import Utils
from jt.utils.misc import read_cfg
from jt.utils.time import datetime2string

DB_CFG_PATH = r'E:\performance_statistics\config'

def init_db_info():    
    dbloader = PgSQLLoader()
    db_cfg = read_cfg('cfg/db.ini', package='ps')['attribution']
    dbloader.set_db_config(db_cfg)
    file_list = Utils.get_all_files(DB_CFG_PATH)
    for cfg in file_list:
        table_name = os.path.basename(cfg)
        table_name = table_name[0: len(table_name)-4]
        print(table_name)
        df = pd.read_csv(cfg, encoding='gbk', dtype=str)
        keys_ = [df.columns.tolist()[0], 'update_time']
        df['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dbloader.upsert(table_name, df, keys_=keys_)
        

if __name__ == "__main__":
    init_db_info()