import os
from jt.utils.calendar.api_calendar import TradeCalendarDB
from jt.utils.fs.utils import Utils as fsutils
from jt.utils.misc.log import Logger

calendardb = TradeCalendarDB()
TODAY = calendardb.get_trading_date() # get T0
_logger = Logger(module_name_=__name__)

def decompress_7Z(filePath_, save_dir_ = None):    
    """
    decompress 7z files
    input: filepath_ : full path of 7z files
    _7z_path_ : .exe of 7z
    save_dir_ : save path
    """ 
    _dir, _name = fsutils.extract_file_dir_and_name(filePath_)
    save_dir_ = _dir if save_dir_ is None else save_dir_
    # os.system(f'decompress.bat {_root} {_7z_path_} {_dir} {_name} {save_dir_}')
    os.system(f'7z x {filePath_} -o{save_dir_}')

def save_result(df_, db, table_name, keys_):    
    _logger.info(f'Save result to DB {table_name}')
    db.upsert(table_name, df_=df_, keys_=keys_)  

def attach_security_type(df_):
    df_copy = df_.copy(deep=True)
    df_copy['security_type'] = df_copy['symbol'].apply(get_security_type)
    return df_copy

def get_security_type(symbol_):    
    _sec_type_default = 'UNKNOWN'
    _sec_type_stock = 'STOCK'
    _sec_type_stock_hk = 'HK'
    _sec_type_fund = 'FUND'
    _sec_type_oc = 'OC'
    _sec_type_repo = 'REPO'
    _sec_type_fx = 'FX'
    _sec_type_bond = 'BOND'
    _sec_type_index = 'INDEX'
    _sec_type_future = 'FUTURE'
    _sec_type_option = 'OPTION'
    _sec_type_cta = 'CTA'
    
    if symbol_[0:1].isalpha(): # the first char is alphabet, include chinese
        # index future
        if symbol_.startswith('IC') or symbol_.startswith('IF') or symbol_.startswith('IH') or \
            (symbol_.startswith('CUSF') and symbol_.endswith('HK')):
            return _sec_type_future   
        # cta
        elif (symbol_.endswith('.CZC') or symbol_.endswith('.SHF') or symbol_.endswith('.DCE') or symbol_.endswith('.INE')) \
            and ('-' not in symbol_) and (len(symbol_) < 11):
            return _sec_type_cta
        # index swap -> fund
        elif (symbol_.startswith('H') or symbol_.startswith('N')) and (symbol_.endswith('.CSI')):
            return _sec_type_fund
        # option
        else:
            return _sec_type_option
    else:        
        # repo
        if symbol_.startswith('204') or (symbol_.startswith('88888')):
            return _sec_type_repo
        # stock
        if (symbol_.startswith('0')) and (symbol_.endswith('.SZ')) or \
                (symbol_.startswith('3')) and (symbol_.endswith('.SZ')) or \
                (symbol_.startswith('6') and symbol_.endswith('.SH')):
            return _sec_type_stock
        # hk
        if (symbol_.startswith('CUSF')) and (symbol_.endswith('.HK')):  # CUSF1906.HK
            return _sec_type_fx
        elif symbol_.endswith('.HK'):
            return _sec_type_stock_hk        
        # bond
        if ((symbol_.startswith('1')) and (symbol_.endswith('.SH')) or \
                (symbol_.startswith('1')) and (symbol_.endswith('.SZ'))) and (len(symbol_)<=9):
            return _sec_type_bond
        # H00905.CSI, N00905.CSI
        if (symbol_.startswith('H') or symbol_.startswith('N')) and (symbol_.endswith('.CSI')):
            return _sec_type_fund # _sec_type_index
        # fund
        if (symbol_.endswith('.OF')) or \
                (symbol_.startswith('51') and symbol_.endswith('.SH')):
            return _sec_type_fund
        # oc
        if symbol_.endswith('.OC'):
            return _sec_type_oc
        # option
        if (len(symbol_) >= 11):
            return _sec_type_option

    _logger.info(f'unknown type in attach_security_type for {symbol_}')
    return _sec_type_default


if __name__ == "__main__":
    d_lists = calendardb.get_trading_calendar(from_='20190822', to_='20190823')
    for d in d_lists:        
        for i in [1,2]:
            file_name_ = f'{d}-KLine-{i}.7z'
            fsutils.move_file(file_name_, from_=r'\\192.168.1.136\data\Wind\tdb\tdb-data-gx\2019', to_=r'D:\temp', replace_=True)
            decompress_7Z(os.path.join(r'D:\temp',file_name_), save_dir_=r'\\192.168.1.88\"Trading Share"\daily_quote\KLine')
    
    pass

    


    