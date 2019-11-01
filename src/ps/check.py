import numpy as np
import pandas as pd

from jt.utils.misc import Logger
from jt.utils.calendar import TradeCalendarDB
from jt.utils.db import PgSQLLoader
from ps.data_loader import get_alloction

calendar = TradeCalendarDB()
today = calendar.get_trading_date()
qidb = PgSQLLoader('qi-account')
attdb = PgSQLLoader('attribution')
_log = Logger(module_name_=__name__)

def check_allocation(date_ = today):
    '''
    check if there is any strategy not been allocated
    '''
    sql = f'''
        SELECT distinct b.strategy_id FROM "position" a, "product_allocation" b where a."date" = '{date_}' and a.account_id=b.strategy_id
        union
        SELECT distinct b.strategy_id FROM "transaction" a, "product_allocation" b where a."date" = '{date_}' and a.account_id=b.strategy_id
    '''
    qi_strategy = qidb.read(sql)
    att_strategy = get_alloction()
    att_strategy = att_strategy.loc[:,'strategy_id'].drop_duplicates()
    ret = pd.merge(qi_strategy, att_strategy, on='strategy_id', how='outer', indicator=True)
    if 'left_only' in set(ret['_merge']):
        _log.info("Strategy don't exist in attribution:")
        _log.info(ret.loc[ret['_merge']=='left_only', 'strategy_id'])

    if 'right_only' in set(ret['_merge']):
        _log.info("Strategy only exist in attribution:")
        _log.info(ret.loc[(ret['_merge']=='right_only') & (~ret['strategy_id'].str.contains('OTHER')) & (~ret['strategy_id'].str.contains('HEDGE')),
        'strategy_id'])

if __name__ == "__main__":
    check_allocation('20191030')