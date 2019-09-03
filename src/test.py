from ps.calculation import daily_statistics
from ps.init_data import download_daily_data
from jt.utils.calendar import TradeCalendarDB
from jt.utils.misc.log import Logger

_logger = Logger(module_name_=__name__)

if __name__ == "__main__":
    calendar = TradeCalendarDB()
    date_list = calendar.get_trading_calendar(from_='20190829', to_='20190903')
    for d in date_list:
        _logger.info(f'Deal date {d}')
        download_daily_data(date_=d, is_overwrite_=False)
        daily_statistics(d)