import pandas as pd

from ps.calculation import daily_statistics
from ps.summary import cal_monthly_performance
from ps.init_data import download_daily_data, init_db_info
from jt.utils.calendar import TradeCalendarDB
from jt.utils.misc.log import Logger

_logger = Logger(module_name_=__name__)

if __name__ == "__main__":
    calendar = TradeCalendarDB()
    date_list = calendar.get_trading_calendar(from_='20200226', to_='20200310')
    for d in date_list:
        _logger.info(f'Deal date {d}')
        download_daily_data(date_=d, is_overwrite_ = False)
        # daily_statistics(d,['98B_ZF502','97A_ZF502'])
    # download_daily_data(date_='20200316', is_overwrite_ = True)
    # daily_statistics('20200206', ['80A_ZS','80A_ARBT'])
    # daily_statistics('20200226')
    
    # init_db_info()

    # bonus, defer_bonus, sr = cal_monthly_performance(from_='20190701', to_='20200131')
    # bonus.to_csv(r'e:\temp\bonus.csv')
    # defer_bonus.to_csv(r'e:\temp\defer_bonus.csv')
    # sr.to_csv(r'e:\temp\sr.csv')
    pass