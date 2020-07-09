import pandas as pd

from ps.calculation import daily_statistics
from ps.summary import cal_monthly_performance
from ps.init_data import download_daily_data, init_db_info
from jt.utils.calendar import TradeCalendarDB
from jt.utils.misc.log import Logger

_logger = Logger(module_name_=__name__)

if __name__ == "__main__":
    # calendar = TradeCalendarDB()
    # date_list = calendar.get_trading_calendar(from_='20200608', to_='20200617')
    # for d in date_list:
        # _logger.info(f'Deal date {d}')
        # download_daily_data(date_=d, is_overwrite_ = False)
        # daily_statistics(d, ['01_ZFM502','105B_HEDGE'])
    # download_daily_data(date_='20200706', is_overwrite_ = True)
    # daily_statistics('20200603')
    # daily_statistics('20200521', new_account=pd.DataFrame({'account_id':['105'], 'totalasset':[320000000]}))
    
    # init_db_info()

    # bonus, defer_bonus, sr = cal_monthly_performance(from_='20190701', to_='20200531')
    # bonus.to_excel(r'e:\temp\bonus_202005.xlsx')
    # defer_bonus.to_excel(r'e:\temp\defer_bonus_202005.xlsx')
    # sr.to_excel(r'e:\temp\sr_202005.xlsx')
    pass