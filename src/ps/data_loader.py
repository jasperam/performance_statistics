import datetime
import numpy as np
import pandas as pd

# from WindPy import w
from copy import copy
from collections import defaultdict

from jt.app.operate.data_loader import DataLoader
from jt.app.operate.gql_loader import GqlRDLWind
from jt.utils.calendar.api_calendar import TradeCalendarDB
from jt.utils.db import PgSQLLoader, SqlServerLoader
from jt.utils.misc import read_cfg
from jt.utils.time.format import string2datetime, datetime2string
from qi.tool.account_loader.utils import attach_security_type

dataloader = DataLoader()
gqlrdlwind = GqlRDLWind()
calendar = TradeCalendarDB()
pgloader = PgSQLLoader('attribution')
# pgloader.set_db_config(read_cfg('cfg/db.ini', package='ps')[])
winddb = SqlServerLoader('ali')
traderdb = SqlServerLoader('trade')

EXCLUDED_SECURITY_LIST = ['204001.SZ']

def get_pos(date_, strategy_ids=None):  
    y_date = calendar.get_trading_date(date_, offset_=-1)
    pos = dataloader.get_position_gql(y_date, y_date, strategy_ids)
    if not pos.empty:
        pos = pos[pos.volume!=0]        
        pos = pos[~pos['symbol'].isin(EXCLUDED_SECURITY_LIST)].reset_index(drop=True)
        pos = attach_security_type(pos)
        pos.rename(columns={'account_id':'strategy_id'}, inplace=True)
    return pos


def get_trade(date_, strategy_ids=None):  
    trade = dataloader.get_transaction_gql(date_, date_, strategy_ids)
    if not trade.empty:
        trade = trade.loc[(trade.volume!=0) & (trade.price!=0)]       
        trade = trade[~trade['symbol'].isin(EXCLUDED_SECURITY_LIST)].reset_index(drop=True)
        trade = attach_security_type(trade)
        trade.rename(columns={'account_id':'strategy_id'}, inplace=True)
    return trade


def get_benchmark_info():   
    sql = """
        SELECT distinct on (bm_id) bm_id,bm_symbol,bm_adjust_return,bm_original_symbol
        FROM "public"."benchmark" order by bm_id,update_time desc
    """    
    return pgloader.read(sql)


def get_alloction():
    def get_product_info():    
        sql = """
            SELECT distinct on (product_id) product_id,product_name,trader,commission
            FROM "public"."product" order by product_id,update_time desc
        """    
        return pgloader.read(sql)    

    def get_strategy_info():   
        sql = """
            SELECT (strategy_id) strategy_id,strategy_name,bm,bm_description,cal_type
            FROM "public"."strategy" order by strategy_id,update_time desc
        """    
        return pgloader.read(sql)

    def get_manager_info():   
        sql = """
            SELECT distinct on (manager_id) manager_id,manager_name 
            FROM "public"."manager" order by manager_id,update_time desc;
        """    
        return pgloader.read(sql)

    sql = """
        SELECT distinct on (strategy_id) strategy_id,product_id,manager_id FROM "public"."allocation" 
        where status='running'
        order by strategy_id,update_time desc;
    """    
    allocation = pgloader.read(sql)

    product = get_product_info()
    strategy = get_strategy_info()   
    manager = get_manager_info()

    allocation = allocation.merge(product, on=['product_id'], how='left')
    allocation = allocation.merge(strategy, on=['strategy_id'], how='left')
    allocation = allocation.merge(manager, on=['manager_id'], how='left')

    return allocation


def get_account_detail(date_):
    return dataloader.get_account_detail(date_)


def set_data(outdata):
    l = list()
    if outdata.ErrorCode!=0:
        print('error code:'+str(outdata.ErrorCode)+'\n')
        return()
    for i in range(0,len(outdata.Data[0])): 
        d = defaultdict(str)
        d['windcode'] = str(outdata.Codes[i])
        if len(outdata.Times)>0:
            d['time'] = str(outdata.Times[i])          
        for k in range(0, len(outdata.Fields)):            
            d[str(outdata.Fields[k]).lower()] = outdata.Data[k][i]           
        l.append(d)
    return l
    

def get_index_wsd(date, codes, factors, w):
    """
    return: dict {windcode: {pct_chg: , close: , pre_close: }}
    """  
    d = defaultdict(str)
    for b in codes:
        wsddata=w.wsd(b, factors, date, date, "Fill=Previous")
        listtemp = set_data(wsddata)
        for i in range(0, len(listtemp)):           
            d[listtemp[i]['windcode']] = listtemp[i]
    return d    
    

def get_index_wsi(code, time, w, factors=["close"]):
    assert isinstance(code, list) 

    def get_next_minutes(time):
        next_minutes = string2datetime(time, "%Y%m%d %H:%M:%S") + datetime.timedelta(minutes=1)
        return datetime2string(next_minutes, "%Y%m%d %H:%M:%S")

    d = defaultdict(str)
    for b in code:
        wsidata=w.wsi(b, factors, time, get_next_minutes(time))
        listtemp = set_data(wsidata)
        for i in range(0, len(listtemp)): 
            d[listtemp[i]['windcode']] = listtemp[i]        
    return d

            
def get_cal_type2_index_wsi(trade_, w):
    """
    get wsi data for index ['000300.SH','000905.SH','000016.SH','000852.SH']
    """        
    index_=['000300.SH','000905.SH','000016.SH','000852.SH'] # 
    date_ = trade_.loc[0, 'date'] # .to_numpy()[0]  
    t_list, p_300_list, p_500_list = list(), list(), list()
    p_50_list, p_1000_list = list(), list()

    # get operating time of trade
    for t_ in trade_.loc[:, 'time'].unique():     
        p_dict = get_index_wsi(index_, date_ + ' ' + t_, w, ['close'])
        t_list = t_list.append(t_)
        p_300_list = p_300_list.append(p_dict['000300.SH']['close'])
        p_500_list = p_500_list.append(p_dict['000905.SH']['close'])
        p_50_list = p_300_list.append(p_dict['000016.SH']['close'])
        p_1000_list = p_500_list.append(p_dict['000852.SH']['close'])

    return pd.DataFrame(index=t_list, 
        data={'000300.SH':p_300_list, '000905.SH':p_500_list, '000016.SH':p_50_list, '000852.SH':p_1000_list})
            
     
def get_prices(symbols, date):
    """
    获取所选代码与日期的昨收/今收
    :param symbols:
    :param date:
    :return:
    """
    return gqlrdlwind.AShareEODPrices(symbols, date, date, 
            response_fileds_=['symbol', 'close', 'pre_close', 'change_rate', 'change_price'], timeout=300)
     

def get_price_dict(date, symbol_df):
    prices_dict = defaultdict(lambda: '')
    prices_dict.update({i['symbol']: i for i in get_prices(list(set(symbol_df)), date)})
    for s in list(set(symbol_df)):
        if s not in prices_dict.keys():
            print(date ,s)
            raise NotImplementedError
    return prices_dict


def get_commission_rate(ACC_INFO):
    def _inner(x): 
        if x not in ACC_INFO.strategy_id.unique().tolist():
            raise Exception(f"{x}'s commission rate is not found!")
        return ACC_INFO.loc[ACC_INFO.strategy_id==x, 'commission'].to_numpy()[0]
    return _inner


def insert_market_data(df, prices):
    """
    数据加工
    :param df:
    :param date:
    :return:
    """
    r = copy(df)
 
    r['close'] = r.apply(lambda rec: prices[rec.symbol]['close'], axis=1)  
    r['pre_close'] = r.apply(lambda rec: prices[rec.symbol]['pre_close'], axis=1)
    r['change_rate'] = r.apply(lambda rec: prices[rec.symbol]['change_rate'], axis=1)  
    r['change_price'] = r.apply(lambda rec: prices[rec.symbol]['change_price'], axis=1)  
    return r


def make_index_minutes_quote(date_, w):
    index_ = ['000016.SH','000300.SH','000905.SH','000852.SH']
    
    pass


def calculate_dvd(date_, pos):
    r = pos[['account_id','symbol','volume']]

    def get_dvd(date_):
        sql = f'''SELECT EX_DT,EQY_RECORD_DT,S_INFO_WINDCODE,STK_DVD_PER_SH,CASH_DVD_PER_SH_AFTER_TAX 
            FROM [dbo].[ASHAREDIVIDEND] where EQY_RECORD_DT='{date_}' and S_DIV_PROGRESS='3' '''
        return winddb.read(sql)

    dvd_record = get_dvd(date_)
    code_list = dvd_record['S_INFO_WINDCODE'].to_list()
    r = r.loc[r['symbol'].isin(code_list)]
    r['dvd_volume_per_share'] = r['symbol'].apply(lambda x: dvd_record.loc[dvd_record['S_INFO_WINDCODE']==x, 'STK_DVD_PER_SH'].to_numpy()[0])
    r['dvd_amount_per_share'] = r['symbol'].apply(lambda x: dvd_record.loc[dvd_record['S_INFO_WINDCODE']==x, 'CASH_DVD_PER_SH_AFTER_TAX'].to_numpy()[0])
    r['ex_dt'] = r['symbol'].apply(lambda x: dvd_record.loc[dvd_record['S_INFO_WINDCODE']==x,'EX_DT'].values[0])
    r['record_dt'] = date_
    r['dvd_volume'] = r['volume'] * r['dvd_volume_per_share']
    r['dvd_amount'] = r['volume'] * r['dvd_amount_per_share']

    r.drop(columns=['dvd_volume_per_share','dvd_amount_per_share'], inplace=True)

    traderdb.upsert('[dbo].[dividend_detail]', df_=r, keys_=['ex_dt','account_id','symbol'])

 
if __name__ == "__main__":
    pos = pd.DataFrame({'account_id': ['12','01','91'],
                        'symbol': ['002129.SZ','000154.SZ','603580.SH'],
                        'volume': [10000, 25000, 32000]})
    calculate_dvd('20190723', pos)



    

    