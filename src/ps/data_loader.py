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
navdb = SqlServerLoader('trade71')

EXCLUDED_SECURITY_LIST = ['204001.SZ'] # the securities that don't deal with

def get_pos(date_, strategy_ids=None):  
    # y_date = calendar.get_trading_date(date_, offset_=nlag)
    pos = dataloader.get_position_gql(date_, date_, strategy_ids)
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
    #FIXME cal the 88 performance
    sql = f'''
        SELECT strategy_name,sum(pnl) as pnl,sum(bm_pnl) as bm_pnl,sum(stock_mv) as stock_mv
        FROM "performance" b 
        where trade_dt between '{from_}' and '{to_}' and strategy_id not in ('86_PR')
        GROUP BY strategy_name
    '''
    ret = attributiondb.read(sql)
    return ret


def get_history_alpha(from_, to_):
    sql = f'''
        select trade_dt,strategy_name,alpha,turnover from (
            select a.trade_dt,strategy_name,round(sum(alpha*abs(stock_mv))/sum(abs(stock_mv)) ,2) as alpha, round(sum(a.stock_turnover/2*stock_mv)/sum(stock_mv) ,2) as turnover
            from "performance" a, "classcification_detail" b where a.stock_mv!=0 and a.trade_dt=b.trade_dt and a.strategy_id=b.strategy_id 
            and (b.stock_pos_counts>=7 or b.hk_stock_pos_counts>0 or a.strategy_name in ('RQ'))  
            and a.trade_dt between '{from_}' and '{to_}' and abs(a.trade_net)/a.stock_mv<=0.5
            GROUP BY a.trade_dt,strategy_name
            union
            select a.trade_dt,strategy_name,round(avg(alpha),2) as alpha,0 as turnover
            from "performance" a, "classcification_detail" b 
            where a.stock_mv=0 and a.trade_dt=b.trade_dt and a.strategy_id=b.strategy_id
            and ((abs(b.cta_pos_pnl)+abs(b.cta_trade_pnl)!=0) or (abs(b.future_pos_pnl)+abs(b.future_trade_pnl)!=0) or (abs(b.option_pos_pnl)+abs(b.option_trade_pnl)!=0))
            and a.trade_dt between '{from_}' and '{to_}'
            GROUP BY a.trade_dt,strategy_name) as foo
        order by strategy_name,trade_dt
    '''
    ret = attributiondb.read(sql)
    return ret


def get_daily_alpha(date_):
    # TODO: market value weighted depended on yesterday mv, not today
    sql = f'''
        select trade_dt,strategy_name,alpha,turnover from (
            select a.trade_dt,strategy_name,round(sum(alpha*abs(stock_mv))/sum(abs(stock_mv)) ,2) as alpha, round(sum(a.stock_turnover/2*stock_mv)/sum(stock_mv) ,2) as turnover
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


def get_dividend_amount_quantity(date_, acc_lists, pos):
    # get dvd amount
    acc_lists_str = "'"+"','".join(acc_lists)+"'"
    sql = f'''
        SELECT strategy_id, sum(dvd_amount) as dvd_amount FROM "public"."dividend_detail" where ex_dt='{date_}' and strategy_id in ({acc_lists_str}) group by strategy_id
    '''
    pos_dvd = attributiondb.read(sql)
    if pos_dvd.empty:
        pos_dvd = pd.DataFrame(columns=['strategy_id', 'dvd_amount'])        

    # get dvd quantity and deal the pos
    sql = f'''
        select strategy_id, symbol, dvd_volume from "public"."dividend_detail" where ex_dt='{date_}' and strategy_id in ({acc_lists_str})
    '''
    pos_dvd_amount = attributiondb.read(sql)
    if pos_dvd_amount.empty:
        pass      
    else:
        pos = pos.merge(pos_dvd_amount, on=['strategy_id','symbol'], how='left').fillna(0)
        pos['volume'] = pos['volume'] + pos['dvd_volume']
        pos.drop(columns='dvd_volume', inplace=True) 
    
    return pos_dvd, pos


def decrease_hk_fee(from_, to_):
    """
    Decrease the cost of some products
    input: from_ date, to_ date
    strategy_ids
    get long position fee from nav db, decrease depend on the mv of strateties
    """
    # get long pos fee
    sql = f"""
        SELECT account_id, sum(-1*long_fee) as total_fee 
        FROM [dbo].[JasperHKPnl] where [date] between '{from_}' and '{to_}' 
        GROUP BY account_id
    """
    fee = navdb.read(sql)

    alloc = get_alloction()
    alloc = alloc.loc[:, ['strategy_id','root_product_id','strategy_name']].drop_duplicates()

    hk_sta_list = alloc.loc[(alloc['root_product_id'].isin(fee['account_id'])) \
        & (~alloc['strategy_id'].str.contains('HEDGE')) & (~alloc['strategy_id'].str.contains('OTHER')), 'strategy_id'].to_list()
    hk_sta_str = "','".join(hk_sta_list)

    # get strategy ratio
    sql = f"""
        SELECT strategy_id, sum(stock_mv) as mv FROM "performance" 
        where trade_dt between '{from_}' and '{to_}' and strategy_id in ('{hk_sta_str}') GROUP BY strategy_id
    """
    stra_mv = attributiondb.read(sql)
    stra_mv = stra_mv.merge(alloc, on='strategy_id', how='left')
    stra_mv['pnl'] = 0
    for p_id in fee['account_id']:
        stra_mv.loc[stra_mv['root_product_id']==p_id, 'pnl'] = stra_mv.loc[stra_mv['root_product_id']==p_id, 'mv']. \
            div(stra_mv.loc[stra_mv['root_product_id']==p_id, 'mv'].sum()). \
            mul(fee.loc[fee['account_id']==p_id, 'total_fee'].to_numpy()[0])
    

    stra_mv['trade_dt'] = to_
    attributiondb.upsert('"public"."one_time_deduction"', stra_mv[['trade_dt','strategy_id','pnl','strategy_name']], 
        keys_=['trade_dt','strategy_id','strategy_name'])
    pass    


if __name__ == "__main__":
    # p = get_pos('20200729','101A_SEO')
    # print(p)
    # decrease_hk_fee('20200701','20200731')
    pass