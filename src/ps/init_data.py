import pandas as pd
import numpy as np
import os
import datetime

from WindPy import w
from collections import defaultdict

from jt.utils.db import PgSQLLoader, SqlServerLoader
from jt.utils.fs import Utils as fsutils
from jt.utils.misc import read_cfg, read_yaml
from jt.utils.time import datetime2string
from jt.utils.calendar import TradeCalendarDB
from jt.utils.time.format import string2datetime, datetime2string
from ps.utils import decompress_7Z

winddb = SqlServerLoader('ali')
tradedb = SqlServerLoader('trade')
calendar = TradeCalendarDB()
TODAY = calendar.get_trading_date()
CONFIG = read_yaml('cfg/file.yaml', package='ps')


def set_data(outdata):
    d = defaultdict(list)
    if outdata.ErrorCode!=0:
        print('error code:'+str(outdata.ErrorCode)+'\n')
        return()
    for i in range(0,len(outdata.Data[0])):         
        if len(outdata.Times) > 1:
            d['time'].append(str(outdata.Times[i]))
        for k in range(0, len(outdata.Fields)):            
            d[str(outdata.Fields[k]).lower()].append(outdata.Data[k][i])
    return d

def init_db_info():    
    """
    upsert configs of PS to DB
    """
    dbloader = PgSQLLoader('attribution')   
    file_list = fsutils.get_all_files(CONFIG.get('DB_CFG', NameError))
    for cfg in file_list:
        table_name = os.path.basename(cfg)
        table_name = table_name[0: len(table_name)-4]
        print(table_name)
        df = pd.read_csv(cfg, encoding='gbk', dtype=str)        
        if table_name=='product':
            keys_=['product_id','sec_type', 'update_time']
        else:
            keys_ = [df.columns.tolist()[0], 'update_time']
        df['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dbloader.upsert(table_name, df, keys_=keys_)
        

def download_ashare_daily_data(w, date_ = TODAY, is_overwrite_ = False):
    """
    Download daily date
    1. index 2. stock 3. index future 4. option 5. bond 6. cta 7. kline 8. fund 9. forex
    """    
    _root = CONFIG.get('DAILY_QUOTE', NotImplementedError)
    y_date_ = calendar.get_trading_date(date_=date_, offset_=-1)      
    cnt_col = ['index_cnt','stock_cnt','index_future_cnt','option_cnt','bond_cnt','cta_cnt','fund_cnt','forex_cnt']
    cnt = pd.DataFrame(np.zeros((1,len(cnt_col))), columns=cnt_col)
    
    # index daily quote
    _quote_file_index = os.path.join(_root, f'index_{date_}.csv')
    if (not fsutils.is_file(_quote_file_index)) or (fsutils.is_file(_quote_file_index) and is_overwrite_):
        BENCHMARK = ['H00016.SH','H00300.CSI','H00905.CSI','H00852.SH']      
        sql = f'''
            SELECT S_INFO_WINDCODE as symbol,S_DQ_PRECLOSE as pre_close,S_DQ_CLOSE as [close],
                    S_DQ_CHANGE as change_price,S_DQ_PCTCHANGE as change_rate
            FROM [dbo].[AINDEXEODPRICES] 
            WHERE S_INFO_WINDCODE in ({"'"+"','".join(BENCHMARK)+"'"}) and TRADE_DT='{date_}'
        '''
        df = winddb.read(sql)        
        if not df.empty:
            df.to_csv(_quote_file_index, encoding='gbk', index=0)
            cnt['index_cnt']=df.shape[0]

    # A share daily quote
    _quote_file_stock = os.path.join(_root, f'stock_{date_}.csv')
    if (not fsutils.is_file(_quote_file_stock)) or (fsutils.is_file(_quote_file_stock) and is_overwrite_):     
        sql = f'''            
            SELECT S_INFO_WINDCODE as symbol,S_DQ_PRECLOSE as pre_close,S_DQ_CLOSE as [close],S_DQ_CHANGE as change_price,
                    S_DQ_PCTCHANGE as change_rate,S_DQ_VOLUME as volume,S_DQ_AMOUNT as amount,S_DQ_AVGPRICE as avg_price,S_DQ_TRADESTATUS as trade_status
            FROM [dbo].[ASHAREEODPRICES] 
            where TRADE_DT='{date_}'
        '''
        df = winddb.read(sql)
        if not df.empty:
            df.to_csv(_quote_file_stock, encoding='gbk', index=0)
            cnt['stock_cnt']=df.shape[0]
    
    # index future daily quote
    _quote_file_future = os.path.join(_root, f'future_{date_}.csv')
    if (not fsutils.is_file(_quote_file_future)) or (fsutils.is_file(_quote_file_future) and is_overwrite_):     
        sql = f'''            
            SELECT a.S_INFO_WINDCODE as symbol,S_DQ_PRESETTLE as pre_settle,S_DQ_SETTLE as settle,b.s_dq_close as pre_close,
                    a.S_DQ_CLOSE as [close],S_DQ_SETTLE as settle,S_DQ_VOLUME as volume,S_DQ_AMOUNT as amount,S_DQ_OI as oi
            FROM [dbo].[CINDEXFUTURESEODPRICES] as a 
            join (select s_info_windcode, s_dq_close from [dbo].[CINDEXFUTURESEODPRICES] where TRADE_DT='{y_date_}') as b 
            on a.TRADE_DT='{date_}' and a.S_INFO_WINDCODE=b.s_info_windcode
        '''
        df = winddb.read(sql)
        if not df.empty:
            df.to_csv(_quote_file_future, encoding='gbk', index=0)
            cnt['index_future_cnt']=df.shape[0]

    # option daily quote    
    _quote_file_option = os.path.join(_root, f'option_{date_}.csv')
    if (not fsutils.is_file(_quote_file_option)) or (fsutils.is_file(_quote_file_option) and is_overwrite_):
        sql = f'''
            SELECT distinct WindCode FROM [dbo].[JasperPosition] where trade_dt = '{y_date_}' and type='OPTION'
            union
            SELECT distinct WindCode FROM [dbo].[JasperTradeDetail] where trade_dt = '{date_}' and type='OPTION'
        '''
        l_symbols = tradedb.read(sql)['WindCode'].values.tolist()
        if l_symbols:
            ret = set_data(w.wss(f"{','.join(l_symbols)}","windcode,sec_name,pre_close,open,high,low,close,pre_settle,settle,chg,pct_chg,volume,amt,dealnum",
                                f"tradeDate={date_};priceAdj=U;cycle=D"))        
            df = pd.DataFrame.from_dict(ret).fillna(0)
            df.rename(columns={'windcode':'symbol', 'sec_name':'name', 'chg':'change_price', 'pct_chg':'change_rate', 'amt':'amount'}, inplace=True)
            df.to_csv(_quote_file_option, encoding='gbk', index=0)
            cnt['option_cnt']=df.shape[0]

    # bond daily quote    
    _quote_file_bond = os.path.join(_root, f'bond_{date_}.csv')
    if (not fsutils.is_file(_quote_file_bond)) or (fsutils.is_file(_quote_file_bond) and is_overwrite_):
        sql = f'''
            SELECT distinct WindCode FROM [dbo].[JasperPosition] where trade_dt = '{y_date_}' and type='B'
            union
            SELECT distinct WindCode FROM [dbo].[JasperTradeDetail] where trade_dt = '{date_}' and type='B'
        '''
        l_symbols = tradedb.read(sql)['WindCode'].values.tolist()
        if l_symbols:
            ret = set_data(w.wss(f"{','.join(l_symbols)}","windcode,pre_close,open,close,volume,amt,chg,pct_chg,trade_status",f"tradeDate={date_};priceAdj=U;cycle=D"))      
            ret['symbol'] = l_symbols
            df = pd.DataFrame.from_dict(ret).fillna(0)
            df.rename(columns={'amt':'amount', 'chg':'change_price', 'pct_chg':'change_rate'}, inplace=True)
            df.to_csv(_quote_file_bond, encoding='gbk', index=0)
            cnt['bond_cnt']=df.shape[0]
    
    # cta daily quote
    _quote_file_cta = os.path.join(_root, f'cta_{date_}.csv')
    if (not fsutils.is_file(_quote_file_cta)) or (fsutils.is_file(_quote_file_cta) and is_overwrite_):        
        sql = f'''
            SELECT distinct WindCode FROM [dbo].[JasperPosition] where trade_dt = '{y_date_}' and type='CTA'
            union
            SELECT distinct WindCode FROM [dbo].[JasperTradeDetail] where trade_dt = '{date_}' and type='CTA'
        '''
        l_symbols = tradedb.read(sql)['WindCode'].values.tolist()   
        if l_symbols:
            ret = set_data(w.wss(f"{','.join(l_symbols)}","windcode,pre_close,open,close,volume,amt,chg,pct_chg",f"tradeDate={date_};priceAdj=U;cycle=D"))        
            ret['symbol'] = l_symbols
            df = pd.DataFrame.from_dict(ret).fillna(0) 
            df.rename(columns={'amt':'amount', 'chg':'change_price', 'pct_chg':'change_rate'}, inplace=True)
            df.to_csv(_quote_file_cta, index=0)
            cnt['cta_cnt']=df.shape[0]
        
    # fund daily quote
    _quote_file_fund = os.path.join(_root, f'fund_{date_}.csv')
    if (not fsutils.is_file(_quote_file_fund)) or (fsutils.is_file(_quote_file_fund) and is_overwrite_):
        sql = f'''
            SELECT distinct WindCode FROM [dbo].[JasperPosition] where trade_dt = '{y_date_}' and type='F'
            union
            SELECT distinct WindCode FROM [dbo].[JasperTradeDetail] where trade_dt = '{date_}' and type='F'
        '''
        l_symbols = tradedb.read(sql)['WindCode'].values.tolist()   
        if l_symbols:
            ret = set_data(w.wss(f"{','.join(l_symbols)}","windcode,pre_close,open,close,volume,amt,chg,pct_chg,trade_status",f"tradeDate={date_};priceAdj=U;cycle=D"))
            ret['symbol'] = l_symbols            
            df = pd.DataFrame.from_dict(ret).fillna(0) 
            df.rename(columns={'amt':'amount', 'chg':'change_price', 'pct_chg':'change_rate'}, inplace=True)
            df.to_csv(_quote_file_fund, encoding='gbk', index=0)
            cnt['fund_cnt']=df.shape[0]

    # forex 
    _quote_file_forex = os.path.join(_root, f'forex_{date_}.csv')
    if (not fsutils.is_file(_quote_file_forex)) or (fsutils.is_file(_quote_file_forex) and is_overwrite_):        
        BENCHMARK = ['HKDCNY.EX','USDCNY.EX']
        ret = set_data(w.wss(BENCHMARK,"windcode,close",f"tradeDate={date_};priceAdj=U;cycle=D"))
        if ret:
            df = pd.DataFrame.from_dict(ret)
            df.rename(columns={'windcode':'symbol'}, inplace=True)   
            df.to_csv(_quote_file_forex, index=0)
            cnt['forex_cnt']=df.shape[0]

    return cnt

def download_hk_daily_data(w, date_ = TODAY, is_overwrite_ = False):
    # HK stock daily quote
    _root = CONFIG.get('DAILY_QUOTE', NotImplementedError)
    cnt_col = ['hkstock_cnt','hkindex_cnt']
    cnt = pd.DataFrame(np.zeros((1,len(cnt_col))), columns=cnt_col)

    _quote_file_hks = os.path.join(_root, f'hkstock_{date_}.csv')
    if (not fsutils.is_file(_quote_file_hks)) or (fsutils.is_file(_quote_file_hks) and is_overwrite_):
        y_date_ = calendar.get_trading_date(date_=date_, offset_=-1)        
        sql = f'''
            SELECT distinct WindCode FROM [dbo].[JasperPosition] where trade_dt = '{y_date_}' and type='HKS'
            union
            SELECT distinct WindCode FROM [dbo].[JasperTradeDetail] where trade_dt = '{date_}' and type='HKS'
        '''
        l_symbols = tradedb.read(sql)['WindCode'].values.tolist()
        if l_symbols:
            ret = set_data(w.wss(f"{','.join(l_symbols)}","windcode,pre_close,open,close,volume,amt,chg,pct_chg,trade_status",f"tradeDate={date_};priceAdj=U;cycle=D"))
            df = pd.DataFrame.from_dict(ret)
            df.rename(columns={'windcode':'symbol', 'amt':'amount', 'chg':'change_price', 'pct_chg':'change_rate'}, inplace=True)
            df.to_csv(_quote_file_hks, encoding='gbk', index=0)
            cnt['hkstock_cnt']=df.shape[0]

    # get other market index
    _quote_file_hkindex = os.path.join(_root, f'hkindex_{date_}.csv')    
    if (not fsutils.is_file(_quote_file_hkindex)) or (fsutils.is_file(_quote_file_hkindex) and is_overwrite_):
        BENCHMARK = ['HSHKI.HI']
        ret = set_data(w.wss(f"{','.join(BENCHMARK)}","windcode,pre_close,close,chg,pct_chg",f"tradeDate={date_};priceAdj=U;cycle=D"))
        if ret:
            df = pd.DataFrame.from_dict(ret).fillna(0)
            df.rename(columns={'windcode':'symbol', 'chg':'change_price', 'pct_chg':'change_rate'}, inplace=True)
            df.to_csv(_quote_file_hkindex, index=0)
            cnt['hkindex_cnt']=df.shape[0]

    return cnt

def download_daily_data(date_ = TODAY, is_overwrite_ = False):
    w.start()
    r = pd.DataFrame()
    if calendar.is_trading_date(date_):
        temp_r = download_ashare_daily_data(w, date_, is_overwrite_)
        r=pd.concat([r,temp_r], axis=1)

    if calendar.is_trading_date(date_, exchange_='hk'):
        temp_r = download_hk_daily_data(w, date_, is_overwrite_)
        r=pd.concat([r,temp_r], axis=1)
    
    w.close()
    return r


def copy_extract_kline_files(date_=TODAY, is_overwrite_=False):
    # decompress kline data    
    for i in [1,2]:
        file_name_ = f'{date_}-KLine-{i}.7z'       
        fsutils.move_file(file_name_, from_=r'\\192.168.1.136\data\Wind\tdb\tdb-data-gx\2019', to_=r'D:\temp', replace_=True)
        _kline_files = os.path.join(r'D:\temp',file_name_)
        if (not fsutils.is_file(_kline_files)) or (fsutils.is_file(_kline_files) and is_overwrite_):
            decompress_7Z(_kline_files, save_dir_=r'\\192.168.1.88\"Trading Share"\daily_quote\KLine')


if __name__ == "__main__":
    # init_db_info()  
    # download_daily_data()  
    # date_list = calendar.get_trading_calendar(from_='20190913', to_='20190911')
    # for d in date_list:
    #     download_daily_data(d)
    # pass
    download_daily_data('20190918')