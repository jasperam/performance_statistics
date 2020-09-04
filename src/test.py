import pandas as pd

from ps.calculation import daily_statistics
from ps.summary import cal_monthly_performance, save_monthly_stat
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
    # download_daily_data(date_='20200812', is_overwrite_ = True)
    # daily_statistics('20200818',['01_ZFM502','105A_SEO'])
    daily_statistics('20200827', to_db=False)
    # daily_statistics('20200817', new_account=pd.DataFrame({'account_id':['58'], 'totalasset':[2000]}))
    
    # init_db_info()

    # decrease_hk_fee('20200701','20200731')
    # bonus, defer_bonus, sr = cal_monthly_performance(from_='20190701', to_='20200731')
    # bonus.to_excel(r'e:\temp\bonus_202007.xlsx')
    # defer_bonus.to_excel(r'e:\temp\defer_bonus_202007.xlsx')
    # sr.to_excel(r'e:\temp\sr_202007.xlsx')
    # save_monthly_stat('20190701','20200630')
    pass