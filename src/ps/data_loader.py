import datetime
import numpy as np
import pandas as pd
import os

# from WindPy import w
from copy import copy
from collections import defaultdict

from jt.app.operate.data_loader import DataLoader
from jt.app.operate.gql_loader import GqlRDLWind
from jt.utils.calendar.api_calendar import TradeCalendarDB
from jt.utils.db import PgSQLLoader, SqlServerLoader
from jt.utils.misc import read_cfg, read_yaml

from ps.utils import attach_security_type

dataloader = DataLoader()
gqlrdlwind = GqlRDLWind()
calendar = TradeCalendarDB()
attributiondb = PgSQLLoader('attribution')
# attributiondb.set_db_config(read_cfg('cfg/db.ini', package='ps')[])
winddb = SqlServerLoader('ali')
traderdb = SqlServerLoader('trade')
CONFIG = read_yaml('cfg/file.yaml', package='ps')

EXCLUDED_SECURITY_LIST = ['204001.SZ'] # the securities that don't deal with

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
    trade = dataloader.get_transaction_gql(date_, date_, strategy_ids,
            fields=['account_id', 'symbol', 'volume', 'price'])
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
    return attributiondb.read(sql)

def get_product_info():    
    sql = """
        SELECT distinct on (product_id, sec_type) product_id,product_name,trader,commission,sec_type,institution,min_commission,root_product_id
        FROM "public"."product" where status='running' order by product_id,sec_type,update_time desc
    """    
    return attributiondb.read(sql)    

def get_strategy_info():   
    sql = """
        SELECT distinct on (strategy_name) strategy_name,bm,bm_description,cal_type,adjust_return,manager_id
        FROM "public"."strategy" order by strategy_name,update_time desc
    """    
    return attributiondb.read(sql)

def get_manager_info():   
    sql = """
        SELECT distinct on (manager_id) manager_id,manager_name,dividend_type
        FROM "public"."manager" order by manager_id,update_time desc
    """    
    return attributiondb.read(sql)


def get_alloction():
    sql = """
        SELECT distinct on (strategy_id) strategy_id,strategy_name,product_id FROM "public"."allocation"   
        where status='running'      
        order by strategy_id,update_time desc;
    """    
    allocation = attributiondb.read(sql)

    product = get_product_info()
    strategy = get_strategy_info()   
    manager = get_manager_info()

    allocation = allocation.merge(product, on=['product_id'], how='left')
    allocation = allocation.merge(strategy, on=['strategy_name'], how='left')
    allocation = allocation.merge(manager, on=['manager_id'], how='left')

    return allocation


def get_account_detail(date_, new_account=None):
    ret = dataloader.get_account_detail(date_)
    if not new_account is None:
        ret = ret.append(new_account, ignore_index=True)
    return ret

     
def get_prices(symbols, date):
    """
    获取所选代码与日期的昨收/今收
    :param symbols:
    :param date:
    :return:
    """
    return gqlrdlwind.AShareEODPrices(symbols, date, date, 
            response_fileds_=['symbol', 'close', 'pre_close', 'change_rate', 'change_price'], timeout=300)
     

def get_daily_quote(date_, type_, encoding_='utf8'): 
    """
    read quote files, return df
    """
    _root = CONFIG.get('DAILY_QUOTE', NotImplementedError)   
    ret = pd.read_csv(os.path.join(_root, f'{type_.lower()}_{date_}.csv'), encoding=encoding_)
    return ret


def get_sc_members(date_, encoding_='utf8'):
    """
    get sc members
    """
    _root = CONFIG.get('PROD', NotImplementedError)
    ret = pd.read_csv(os.path.join(_root, f'{date_}.shsz_sc_members.csv'), encoding=encoding_)
    return ret


def get_commission_rate(ACC_INFO, security_type_):
    def _inner(x): 
        if x['strategy_id'] not in ACC_INFO.strategy_id.unique().tolist():
            raise Exception(f"{x}'s commission rate is not found!")
        return ACC_INFO.loc[(ACC_INFO.strategy_id==x['strategy_id']) & (ACC_INFO.sec_type==security_type_), 'commission'].to_numpy()[0]
    return _inner


def calculate_dvd(date_, pos):
    r = pos[['strategy_id','symbol','volume']]

    def get_dvd(date_):
        sql = f'''SELECT EX_DT,EQY_RECORD_DT,S_INFO_WINDCODE,STK_DVD_PER_SH,CASH_DVD_PER_SH_AFTER_TAX 
            FROM [dbo].[ASHAREDIVIDEND] where EQY_RECORD_DT='{date_}' and S_DIV_PROGRESS='3' '''
        return winddb.read(sql)

    dvd_record = get_dvd(date_)
    code_list = dvd_record['S_INFO_WINDCODE'].to_list()
    r = r.loc[r['symbol'].isin(code_list)]
    if not r.empty:
        r['dvd_volume_per_share'] = r['symbol'].apply(lambda x: dvd_record.loc[dvd_record['S_INFO_WINDCODE']==x, 'STK_DVD_PER_SH'].to_numpy()[0])
        r['dvd_amount_per_share'] = r['symbol'].apply(lambda x: dvd_record.loc[dvd_record['S_INFO_WINDCODE']==x, 'CASH_DVD_PER_SH_AFTER_TAX'].to_numpy()[0])
        r['ex_dt'] = r['symbol'].apply(lambda x: dvd_record.loc[dvd_record['S_INFO_WINDCODE']==x,'EX_DT'].values[0])
        r['record_dt'] = date_
        r['dvd_volume'] = r['volume'] * r['dvd_volume_per_share']
        r['dvd_amount'] = r['volume'] * r['dvd_amount_per_share']

        r.drop(columns=['dvd_volume_per_share','dvd_amount_per_share'], inplace=True)    
        attributiondb.upsert('dividend_detail', df_=r, keys_=['ex_dt','strategy_id','symbol'])

    return r


def cal_option_volume(date_):
    sql = f"""
        select account as strategy_id, sum(Qty) as option_trade_volume
        from dbo.JasperTradeDetail 
        where trade_dt='{date_}' and type='OPTION' and not (side=2 and OCtag='open') GROUP BY account;
    """    
    ret = traderdb.read(sql)
    return ret


def cal_cta_commission(date_):
    sql = f"""
        select account as strategy_id, sum(Commission+Fee) as cta_commission 
        from dbo.JasperTradeDetail 
        where trade_dt='{date_}' and type='CTA' GROUP BY account;
    """    
    ret = traderdb.read(sql)
    return ret


def get_deduction(from_, to_):
    sql = f"""
        SELECT strategy_name,sum(pnl) as deduction 
        FROM "one_time_deduction" a
        where trade_dt between '{from_}' and '{to_}'
        group by strategy_name;
    """    
    ret = attributiondb.read(sql)
    return ret


def get_performance(from_, to_):
    sql = f'''
        SELECT strategy_name,sum(pnl) as pnl,sum(bm_pnl) as bm_pnl,sum(stock_mv) as stock_mv
        FROM "performance" b 
        where trade_dt between '{from_}' and '{to_}'
        GROUP BY strategy_name
    '''
    ret = attributiondb.read(sql)
    return ret


def get_history_alpha(from_, to_):
    sql = f'''
        select trade_dt,strategy_name,alpha,turnover from (
            select a.trade_dt,strategy_name,round(sum(alpha*stock_mv)/sum(stock_mv) ,2) as alpha, round(sum(a.stock_turnover/2*stock_mv)/sum(stock_mv) ,2) as turnover
            from "performance" a, "classcification_detail" b where a.stock_mv!=0 and a.trade_dt=b.trade_dt and a.strategy_id=b.strategy_id and (b.stock_pos_counts>=7 or b.hk_stock_pos_counts>0)  
            and a.trade_dt between '{from_}' and '{to_}' and abs(a.trade_net)/a.stock_mv<=0.5
            GROUP BY a.trade_dt,strategy_name
            union
            select trade_dt,strategy_name,round(avg(alpha),2) as alpha,0 as turnover
            from "performance" where stock_mv=0 and trade_dt between '{from_}' and '{to_}'
            GROUP BY trade_dt,strategy_name) as foo
        order by strategy_name,trade_dt
    '''
    ret = attributiondb.read(sql)
    return ret


def get_daily_alpha(date_):
    sql = f'''
        select trade_dt,strategy_name,alpha,turnover from (
            select a.trade_dt,strategy_name,round(sum(alpha*stock_mv)/sum(stock_mv) ,2) as alpha, round(sum(a.stock_turnover/2*stock_mv)/sum(stock_mv) ,2) as turnover
            from "performance" a where a.stock_mv!=0  
            and a.trade_dt = '{date_}'
            GROUP BY a.trade_dt,strategy_name
            union
            select trade_dt,strategy_name,round(avg(alpha),2) as alpha,0 as turnover
            from "performance" where stock_mv=0 and trade_dt = '{date_}'
            GROUP BY trade_dt,strategy_name) as foo
        order by strategy_name,trade_dt
    '''
    ret = attributiondb.read(sql)
    return ret


def get_cumulative_bonus():
    sql = f'''
        select trade_dt,manager_name,cumulative_bonus,bonus,current_redemption,defer
        from "bonus"
    '''
    ret = attributiondb.read(sql)
    return ret


def get_defer_bonus(to_):
    date_ = str(int(to_[0:4])-1)+to_[4:8]
    sql = f'''
        select trade_dt,manager_name,defer as deferred_bonus
        from "bonus" where trade_dt='{date_}' and defer>0
    '''
    ret = attributiondb.read(sql)
    return ret
    

def get_argo_pnl(from_, to_):   
    sql = f'''
        select manager_name as argo_manager,strategy_name,sum(pnl) as argo_pnl
        from "argo_pnl" where trade_dt between '{from_}' and '{to_}'
        group by argo_manager, strategy_name
    '''
    ret = attributiondb.read(sql)
    return ret


if __name__ == "__main__":
    ret = get_defer_bonus('20200930')
    print(ret)