"""
@author: Neo

calculation performance
"""

import os
os.chdir("C:\\Users\\jasper\\Desktop\\performance_statistics-master\\src\\ps")
import pandas as pd
import numpy as np
from ps.data_loader import get_pos, get_trade, get_benchmark_info, get_alloction, \
    get_account_detail, get_daily_quote, get_commission_rate, calculate_dvd, cal_option_volume, \
    get_sc_members, cal_cta_commission, get_dividend_amount_quantity, get_jbm_500
from ps.utils import save_result
from jt.utils.misc.log import Logger
from jt.utils.calendar.api_calendar import TradeCalendarDB
from jt.utils.db import PgSQLLoader
from jt.utils.misc import read_cfg
from jt.utils.time import datetime2string, num_of_week

from jt.utils.misc.read_cfg import read_yaml
from jt.utils.db.loader import DBLoader, SqlServerLoader
from jt.app.business.daily_workflow import at_night_report
from jt.app.business.check_status import *
from jt.app.business.etl_data import *
from jt.app.business.cal_data import cal_modelperformance
from jt.app.operate.data_loader import DataLoader
from jt.app.business.analyse_trade import *
from ps.check import check_allocation

from tqdm import tqdm
from qi.tool.account_loader.constant import jasper_nav_reader
from qi.tool.account_loader.constant import attr_reader

import datetime
import pyautogui

FEE_RATE =  1/1000
CFE_RATE = 0.23/10000

MULTIPLIER_DICT = {
    'IC': 200,
    'IF': 300,
    'IH': 300,
    'CU': 100000
}

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
_sec_type_qfii = 'QFII'
_sec_type_cta = 'CTA'
_sec_type_margin = 'MARGIN'

calendar = TradeCalendarDB()
attributiondb = PgSQLLoader('attribution')
log = Logger(module_name_=__name__)

ALLOCATION = get_alloction()

def get_multiplier():
    def _inner(x):
        if x['security_type'] == _sec_type_future:
            return MULTIPLIER_DICT[str(x['symbol'])[0:2]]    #判断前面2个字符
        else:
            return 1        # stock/fund -> 1
    return _inner

def get_benchmark_rct(BM, date_):
    def _inner(strategy_):
        # print(strategy_)
        f_ = ALLOCATION.loc[ALLOCATION['strategy_id']==strategy_, 'bm'].values[0]
        # judge if the date is the third week
        if num_of_week(date_) == 3 and f_ == '#6': # use IC01.CFE if it's the third week else IC00.CFE
            f_ = '#7'
        for x, y in zip(BM['bm_id'], BM['change_rate'].astype(str)):
            # if isinstance(x, str):
            f_ = f_.replace(x, y)
        return eval(f_)/100
    return _inner

def get_asset(account_detail):
    def _inner(strategy_):
        if (strategy_ == '80_MJOPT') or (strategy_ == '80B_MJOPT'):
            asset = 3000000
        else:
            account_id = ALLOCATION.loc[ALLOCATION['strategy_id']==strategy_, 'root_product_id'].values[0] 
            # print(strategy_)
            asset = account_detail.loc[account_detail['account_id']==account_id, 'totalasset'].values[0]
        return asset
    return _inner

def insert_market_data(df, prices):
    """
    数据加工
    :param df:
    :param date:
    :return:
    """
    df = df.merge(prices[['symbol','close','pre_close','change_price','change_rate','trade_status']], on='symbol', how='left')    
    return df

def cal_pos_return(pos):    
    pos['multiplier'] = 1
    pos['stock_amount'] = pos['close'] * pos['volume'] * pos['multiplier']
    pos['stock_pre_amount'] = pos['pre_close'] * pos['volume'] * pos['multiplier']
    pos['stock_occupy'] = pos['close'] * abs(pos['volume']) * pos['multiplier']    
    pos['stock_pre_occupy'] = pos['pre_close'] * abs(pos['volume']) * pos['multiplier']
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_margin, ['strategy_id','commission','min_commission','margin_rate']]
    pos = pos.merge(_allocation, on='strategy_id', how='left').fillna(0)

    # Simple handling of currency interest rate of 1.7%
    pos['marginfee'] = pos.apply(lambda x: abs(x['stock_amount']) * (x['margin_rate']-0.017) / 365 
                                if (x['volume']<0) and ('_RQ' in x['strategy_id']) else 0, axis=1)

    pos['stock_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier'] - pos['marginfee']
    pos_stats = pos[['strategy_id', 'stock_pos_pnl', 'stock_amount', 'stock_pre_amount', 'marginfee',
                     'stock_occupy','stock_pre_occupy']].groupby(by='strategy_id').sum().reset_index()
    tmp_counts = pos[['strategy_id','date']].groupby('strategy_id').count().reset_index()
    tmp_counts.rename(columns={'date':'stock_pos_counts'}, inplace=True)
    pos_stats = pos_stats.merge(tmp_counts, on='strategy_id', how='inner')
    # 判断资金占用及方向，不改变stock_mv符号
    pos_stats['stock_amount'] = pos_stats.apply(lambda x: x['stock_amount']/abs(x['stock_amount'])*x['stock_occupy'] 
                                                if abs(x['stock_amount'])!=abs(x['stock_occupy']) 
                                                else x['stock_amount'], axis=1)
    pos_stats['stock_pre_amount'] = pos_stats.apply(lambda x: x['stock_pre_amount']/abs(x['stock_pre_amount'])*x['stock_pre_occupy'] 
                                                    if abs(x['stock_pre_amount'])!=abs(x['stock_pre_occupy']) 
                                                    else x['stock_pre_amount'], axis=1)
    return pos_stats.drop(columns=['stock_occupy','stock_pre_occupy'])

def cal_trade_return(trade, sc_member): 
    trade['multiplier'] = 1    
    trade['stock_trade_amount'] = abs(trade['volume'] * trade['price'] * trade['multiplier'])
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_stock, ['strategy_id','commission','min_commission']]
    trade = trade.merge(_allocation, on='strategy_id', how='left')
    # ---------------------------------
    # deal with qfii_fee of prelude(96)
    # qfii_fee_96 = ALLOCATION.loc[(ALLOCATION.sec_type=='QFII') & (ALLOCATION.product_id=='96'), 'commission'].to_numpy()[0]
    # trade.loc[trade['strategy_id'].str.contains('96') & trade['symbol'].isin(sc_member['symbol']), 'commission'] = qfii_fee_96
    # ---------------------------------
    trade['stock_commission'] = trade.apply(lambda x: x['stock_trade_amount']*x['commission'] if x['stock_trade_amount']*x['commission'] 
                                >x['min_commission'] else x['min_commission'], axis=1)
    trade['stock_trade_net_close'] = trade['volume'] * trade['close'] * trade['multiplier']
    trade.loc[trade['volume']<0, 'stock_fee'] = trade['stock_trade_amount'] * FEE_RATE
    trade.loc[trade['volume']>0, 'stock_fee'] = 0
    trade['stock_trade_pnl'] = (trade['close']-trade['price'])*trade['volume']*trade['multiplier'] - trade['stock_fee'] - trade['stock_commission']
    trade['stock_buy'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']>0 else 0, axis=1)     
    trade['stock_sell'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']<0 else 0, axis=1)

    trade_stats = trade[['strategy_id','stock_trade_pnl','stock_fee','stock_trade_net_close','stock_buy','stock_sell','stock_commission']].groupby(
        by='strategy_id').sum().reset_index()       

    return trade_stats

def cal_bond_pos_return(pos):    
    pos['multiplier'] = 1
    pos['bond_amount'] = pos['close'] * pos['volume'] * pos['multiplier']
    pos['bond_pre_amount'] = pos['pre_close'] * pos['volume'] * pos['multiplier']
    pos['bond_occupy'] = pos['close'] * abs(pos['volume']) * pos['multiplier']    
    pos['bond_pre_occupy'] = pos['pre_close'] * abs(pos['volume']) * pos['multiplier']
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_bond, ['strategy_id','commission','min_commission']]
    pos = pos.merge(_allocation, on='strategy_id', how='left').fillna(0)
    pos['bond_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier']
    pos_stats = pos[['strategy_id', 'bond_pos_pnl', 'bond_amount', 'bond_pre_amount', 
                     'bond_occupy','bond_pre_occupy']].groupby(by='strategy_id').sum().reset_index()
    tmp_counts = pos[['strategy_id','date']].groupby('strategy_id').count().reset_index()
    tmp_counts.rename(columns={'date':'bond_pos_counts'}, inplace=True)
    pos_stats = pos_stats.merge(tmp_counts, on='strategy_id', how='inner')
    return pos_stats.drop(columns=['bond_occupy','bond_pre_occupy'])

def cal_bond_trade_return(trade): 
    trade['multiplier'] = 1    
    trade['bond_trade_amount'] = abs(trade['volume'] * trade['price'] * trade['multiplier'])
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_bond, ['strategy_id','commission','min_commission']]
    trade = trade.merge(_allocation, on='strategy_id', how='left')    
    trade['bond_commission'] = trade.apply(lambda x: x['bond_trade_amount']*x['commission'] 
                                           if x['bond_trade_amount']*x['commission']>x['min_commission'] 
                                           else x['min_commission'], axis=1)
    trade['bond_fee'] = 0
    trade['bond_trade_net_close'] = trade['volume'] * trade['close'] * trade['multiplier']
    trade['bond_trade_pnl'] = (trade['close']-trade['price'])*trade['volume']*trade['multiplier'] - trade['bond_fee'] - trade['bond_commission']
    trade['bond_buy'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']>0 else 0, axis=1)     
    trade['bond_sell'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']<0 else 0, axis=1)

    trade_stats = trade[['strategy_id','bond_trade_pnl','bond_fee','bond_trade_net_close','bond_buy','bond_sell','bond_commission']].groupby(
        by='strategy_id').sum().reset_index()       

    return trade_stats

def cal_future_pos_return(pos): 
    pos['multiplier'] = pos.apply(get_multiplier(), axis=1)
    pos['future_amount'] = pos['settle']*pos['volume']*pos['multiplier']
    pos['future_pos_pnl'] = (pos['settle']-pos['pre_settle'])*pos['multiplier']*pos['volume'] 
    return pos[['strategy_id','future_pos_pnl','future_amount']].groupby('strategy_id').sum().reset_index()

def cal_future_trade_return(trade):
    trade['multiplier'] = trade.apply(get_multiplier(), axis=1)
    trade['future_trade_net_close'] = trade['settle']*trade['volume']*trade['multiplier']
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_future, ['strategy_id','commission','min_commission']]
    trade = trade.merge(_allocation, on='strategy_id', how='left')
    trade['future_commission'] = abs(trade['price']*trade['volume']*trade['multiplier'])*trade['commission']*CFE_RATE
    trade['future_trade_pnl'] = (trade['settle']-trade['price'])*trade['multiplier']*trade['volume'] - trade['future_commission']
    trade['future_buy'] = trade.apply(lambda x: x['price']*x['volume']*x['multiplier'] if x['volume']>0 else 0, axis=1)     
    trade['future_sell'] = trade.apply(lambda x: x['price']*x['volume']*x['multiplier'] if x['volume']<0 else 0, axis=1)     
    return trade[['strategy_id','future_trade_pnl','future_commission','future_trade_net_close','future_buy','future_sell']].groupby('strategy_id').sum().reset_index()

def cal_fund_pos_return(pos):    
    pos['multiplier'] = 1
    pos['fund_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier']
    pos['fund_amount'] = pos['close'] * pos['volume'] * pos['multiplier']
    # Deduct Hong Kong product fees
    pos.loc[pos['strategy_id']=='96_HEDGE','fund_pos_pnl'] = pos.loc[pos['strategy_id']=='96_HEDGE','fund_pos_pnl'] - \
        abs(pos.loc[pos['strategy_id']=='96_HEDGE','fund_amount']) * 0.09 / 245
    pos_stats = pos[['strategy_id', 'fund_pos_pnl', 'fund_amount']].groupby(
        by='strategy_id').sum().reset_index()
    return pos_stats

def cal_fund_trade_return(trade):    
    trade['multiplier'] = 1
    trade['fund_trade_amount'] = abs(trade['volume'] * trade['price'] * trade['multiplier'])
    trade['fund_trade_net_close'] = trade['volume'] * trade['close'] * trade['multiplier']
    trade['fund_trade_pnl'] = (trade['close']-trade['price'])*trade['volume']*trade['multiplier']
    trade['fund_buy'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']>0 else 0, axis=1)     
    trade['fund_sell'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']<0 else 0, axis=1)     
    trade_stats = trade[['strategy_id', 'fund_trade_pnl', 'fund_trade_net_close', 'fund_buy', 'fund_sell']].groupby(
        by='strategy_id').sum().reset_index()
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_stock, ['strategy_id','commission']]
    trade_stats = trade_stats.merge(_allocation, on='strategy_id', how='left')
    # trade_stats['fund_commission_rate'] = trade_stats.apply(get_commission_rate(ALLOCATION, _sec_type_stock), axis=1)
    trade_stats['fund_commission'] = trade_stats['commission'] * (trade_stats['fund_buy'] + abs(trade_stats['fund_sell']))
    trade_stats['fund_trade_pnl'] = trade_stats['fund_trade_pnl'] - trade_stats['fund_commission']
    return trade_stats

def cal_hk_pos_return(pos, forex):    
    pos['multiplier'] = 1    
    pos['hk_stock_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier'] * forex
    pos['hk_stock_amount'] = pos['close'] * pos['volume'] * pos['multiplier'] * forex
    pos['hk_stock_pre_amount'] = pos['pre_close'] * pos['volume'] * pos['multiplier'] * forex
    pos_stats = pos[['strategy_id', 'hk_stock_pos_pnl', 'hk_stock_amount','hk_stock_pre_amount']].groupby(
        by='strategy_id').sum().reset_index()
    tmp_counts = pos[['strategy_id','date']].groupby('strategy_id').count().reset_index()
    tmp_counts.rename(columns={'date':'hk_stock_pos_counts'}, inplace=True)
    pos_stats = pos_stats.merge(tmp_counts, on='strategy_id', how='inner')
    pos_stats['hk_stock_forex'] = forex
    return pos_stats

def cal_hk_trade_return(trade, forex):    
    trade['multiplier'] = 1
    trade['hk_stock_trade_amount'] = abs(trade['volume'] * trade['price'] * trade['multiplier'] * forex)
    trade['hk_stock_trade_net_close'] = trade['volume'] * trade['close'] * trade['multiplier'] * forex
    trade['hk_stock_fee'] = trade.apply(lambda x:
        x['hk_stock_trade_amount'] * FEE_RATE if x['volume'] < 0 else 0, axis=1)      
    trade['hk_stock_trade_pnl'] = (trade['close']-trade['price'])*trade['volume']*trade['multiplier'] * forex
    trade['hk_stock_buy'] = trade.apply(lambda x: x['price'] * x['volume'] * x['multiplier'] * forex if x['volume']>0 else 0, axis=1)     
    trade['hk_stock_sell'] = trade.apply(lambda x: x['price'] * x['volume'] * x['multiplier'] * forex if x['volume']<0 else 0, axis=1)     
    trade_stats = trade[['strategy_id', 'hk_stock_trade_pnl', 'hk_stock_fee', 'hk_stock_trade_net_close', 'hk_stock_buy', 'hk_stock_sell']].groupby(
        by='strategy_id').sum().reset_index()
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_stock, ['strategy_id','commission']]
    trade_stats = trade_stats.merge(_allocation, on='strategy_id', how='left')
    trade_stats['hk_stock_commission'] = trade_stats['commission'] * (trade_stats['hk_stock_buy'] + abs(trade_stats['hk_stock_sell']))
    trade_stats['hk_stock_trade_pnl'] = trade_stats['hk_stock_trade_pnl'] - trade_stats['hk_stock_commission'] - trade_stats['hk_stock_fee']
    return trade_stats

def cal_option_pos_return(pos):    
    pos['option_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier']
    pos['option_amount'] = pos['settle'] * pos['volume'] * pos['multiplier']
    pos_stats = pos[['strategy_id', 'option_pos_pnl', 'option_amount']].groupby(
        by='strategy_id').sum().reset_index()
    return pos_stats
    
def cal_option_trade_return(trade, date_):    
    trade['option_trade_net_close'] = trade['volume'] * trade['settle'] * trade['multiplier']
    # trade['option_trade_amount'] = trade['volume'] * trade['price'] * trade['multiplier']  
    trade['option_trade_pnl'] = (trade['settle']-trade['price'])*trade['volume']*trade['multiplier']
    trade['option_buy'] = trade.apply(lambda x: x['price'] * x['volume'] * x['multiplier'] if x['volume']>0 else 0, axis=1)     
    trade['option_sell'] = trade.apply(lambda x: x['price'] * x['volume'] * x['multiplier'] if x['volume']<0 else 0, axis=1)     
    trade_stats = trade[['strategy_id', 'option_trade_pnl', 'option_buy', 'option_sell', 'option_trade_net_close']].groupby(
        by='strategy_id').sum().reset_index()   
    _commission = cal_option_volume(date_) # df[strategy_id, option_trade_volume]
    if _commission.empty:
        trade_stats['option_commission'] = 0
    else:
        _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_option, ['strategy_id','commission']]
        _commission.loc[_commission['strategy_id']=='80B','strategy_id']='80B_MJOPT'
        _commission = _commission.merge(_allocation, on='strategy_id', how='left')
        _commission['option_commission'] = _commission['option_trade_volume'] * _commission['commission']        
        trade_stats = trade_stats.merge(_commission[['strategy_id','option_commission']], on='strategy_id', how='left').fillna(0)    
    trade_stats['option_trade_pnl'] = trade_stats['option_trade_pnl'] - trade_stats['option_commission']
    return trade_stats

def cal_cta_pos_return(pos): 
    pos['cta_amount'] = pos['settle']*pos['volume']*pos['multiplier']
    pos['cta_pos_pnl'] = (pos['settle']-pos['pre_settle']) * pos['volume'] * pos['multiplier']
    return pos[['strategy_id','cta_pos_pnl','cta_amount']].groupby('strategy_id').sum().reset_index()

def cal_cta_trade_return(trade, date_):
    trade['cta_trade_net_close'] = trade['settle']*trade['volume']*trade['multiplier']   
    trade['cta_trade_pnl'] = (trade['settle']-trade['price'])*trade['volume']*trade['multiplier']
    trade['cta_buy'] = trade.apply(lambda x: x['price']*x['volume']*x['multiplier'] if x['volume']>0 else 0, axis=1)     
    trade['cta_sell'] = trade.apply(lambda x: x['price']*x['volume']*x['multiplier'] if x['volume']<0 else 0, axis=1)     
    trade_stats = trade[['strategy_id', 'cta_trade_pnl', 'cta_buy', 'cta_sell', 'cta_trade_net_close']].groupby(
        by='strategy_id').sum().reset_index()
    _commission = cal_cta_commission(date_) # df[strategy_id, option_trade_volume]    
    if _commission.empty:
        trade_stats['cta_commission'] = 0
    else:
        trade_stats = trade_stats.merge(_commission[['strategy_id','cta_commission']], on='strategy_id', how='left').fillna(0)
    trade_stats['cta_trade_pnl'] = trade_stats['cta_trade_pnl'] - trade_stats['cta_commission']
    return trade_stats

def merge_pos_trade_stats(date_, pos_stats, trade_stats, account_detail, BM):
    stats = pd.merge(pos_stats, trade_stats, how='outer', on='strategy_id', suffixes=('_pos','_trade')).fillna(0)
    stats['trade_dt'] = date_
    stats['bm'] = stats['strategy_id'].apply(get_benchmark_rct(BM, date_))
    stats['bm'] = stats['bm'] * 10000

    stats['stock_amount'] = stats['stock_amount'] + stats['stock_trade_net_close'] + \
                            stats['bond_amount'] + stats['bond_trade_net_close']
    stats['hk_stock_amount'] = stats['hk_stock_amount'] + stats['hk_stock_trade_net_close']

    stats['stock_pre_mv'] = stats['stock_pre_amount'] + stats['hk_stock_pre_amount'] + stats['bond_pre_amount']
    stats['stock_mv'] = stats['stock_amount'] + stats['hk_stock_amount'] + stats['bond_amount']
    
    stats['future_pre_amount'] = stats['future_amount']
    stats['future_amount'] = stats['future_amount'] + stats['future_trade_net_close']
    
    stats['fund_pre_amount'] = stats['fund_amount']
    stats['fund_amount'] = stats['fund_amount'] + stats['fund_trade_net_close']    

    stats['option_pre_amount'] = stats['option_amount']
    stats['option_amount'] = stats['option_amount'] + stats['option_trade_net_close']

    stats['cta_pre_amount'] = stats['cta_amount']
    stats['cta_amount'] = stats['cta_amount'] + stats['cta_trade_net_close']

    stats['pos_pnl'] = stats['stock_pos_pnl'] + stats['dvd_amount'] + stats['bond_pos_pnl'] + stats['future_pos_pnl'] + \
        stats['fund_pos_pnl'] + stats['option_pos_pnl'] + stats['hk_stock_pos_pnl'] + stats['cta_pos_pnl']
    stats['trade_pnl'] = stats['stock_trade_pnl'] + stats['bond_trade_pnl'] + stats['future_trade_pnl'] +\
        stats['fund_trade_pnl'] + stats['option_trade_pnl'] + stats['hk_stock_trade_pnl'] + stats['cta_trade_pnl']
    stats['pnl'] = stats['pos_pnl'] + stats['trade_pnl']
    stats['buy'] = stats['stock_buy'] + stats['bond_buy'] + stats['future_buy'] + stats['fund_buy'] + \
        stats['option_buy'] + stats['hk_stock_buy'] + stats['cta_buy']
    stats['sell'] = stats['stock_sell'] + stats['bond_sell'] + stats['future_sell'] + stats['fund_sell'] +\
        stats['option_sell'] + stats['hk_stock_sell'] + stats['cta_sell']
    stats['trade_net'] = stats['buy'] + stats['sell']

    stats['d_pre_amount'] = abs(stats['future_pre_amount'])+abs(stats['option_pre_amount'])+abs(stats['cta_pre_amount'])
    stats['d_amount'] = stats.apply(lambda x: x['d_pre_amount']+\
        max(abs(x['future_buy'])+abs(x['option_buy'])+abs(x['cta_buy']),\
            abs(x['future_sell'])+abs(x['option_sell'])+abs(x['cta_sell'])),axis=1)

    stats['commission'] = stats['stock_commission'] + stats['bond_commission'] + stats['fund_commission'] + \
        stats['hk_stock_commission'] + stats['option_commission'] + stats['future_commission'] + stats['cta_commission']
    stats['fee'] = stats['stock_fee'] + stats['hk_stock_fee'] + stats['bond_fee']
    stats['stock_turnover'] = stats.apply(lambda x: (x['stock_buy'] + abs(x['stock_sell']) + x['bond_buy'] + abs(x['bond_sell'])) / \
                                         (x['stock_pre_amount']+x['bond_pre_amount']+max(x['stock_buy']+x['stock_sell'],0)+\
                                          max(x['bond_buy']+x['bond_sell'],0)) if x['stock_pre_amount'] > 0 else 0 , axis=1)
    stats['hk_stock_turnover'] = stats.apply(lambda x: (x['hk_stock_buy'] + abs(x['hk_stock_sell'])) / x['hk_stock_pre_amount']
        if x['hk_stock_pre_amount'] > 0 else 0 , axis=1)
    stats['product_asset'] = stats['strategy_id'].apply(get_asset(account_detail)) 
    stats['stock_mv_ratio'] = stats['stock_mv'] / stats['product_asset']
    stats['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stats = stats.merge(ALLOCATION[['strategy_id', 'strategy_name']].drop_duplicates(), on='strategy_id', how='left')

    return stats


def cal_type_3(stats):
    """
    cal type 3:
    考虑期货对冲、ETF对冲、指数互换对冲
    """
    # summary in stats, insert into DB
    stats['ret'] = stats.apply(lambda x: \
                x['pnl'] / max(x['d_pre_amount'], x['d_amount']) * 10000 
                if max(x['d_pre_amount'], x['d_amount'])!=0 
                else x['pnl'] / abs(x['fund_pre_amount']+min(0,x['trade_net'])) * 10000, axis=1)
    stats['alpha'] = stats['ret'] - stats['bm']
    stats['pos_ret'] = stats['pos_pnl'] / stats['product_asset'] * 10000 
    stats['trade_ret'] = stats['trade_pnl'] / stats['product_asset'] * 10000
    stats['bm_pnl'] = stats['product_asset'] * stats['bm']  / 10000
    return stats


def cal_type_2(stats):
    if stats.empty:
        stats = pd.concat([stats, pd.DataFrame(columns=['ret', 'alpha', 'pos_ret', 'trade_ret', 'bm_pnl'])], axis=1)
    else:
        stats['ret'] = stats.apply(lambda x: \
            x['pnl']/(x['hk_stock_pre_amount']+max(0,x['stock_buy']+x['stock_sell']))*10000 
            if x['hk_stock_pre_amount']+max(0,x['stock_buy']+x['stock_sell'])>0 else 0, axis=1)
        # stats['ret'] = stats.apply(lambda x: \
        #     x['pnl']/x['hk_stock_pre_amount']*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)
        stats['alpha'] = stats['ret'] - stats['bm']    
        stats['pos_ret'] = stats.apply(lambda x: \
            x['pos_pnl']/(x['hk_stock_pre_amount']+max(0,x['stock_buy']+x['stock_sell']))*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)
        # stats['pos_ret'] = stats.apply(lambda x: \
        #     x['pos_pnl']/x['hk_stock_pre_amount']*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)
        stats['trade_ret'] = stats.apply(lambda x: \
            x['trade_pnl']/(x['hk_stock_pre_amount']+max(0,x['stock_buy']+x['stock_sell']))*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)     
        # stats['trade_ret'] = stats.apply(lambda x: \
        #     x['trade_pnl']/x['hk_stock_pre_amount']*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)
        stats['bm_pnl'] = stats.apply(lambda x: \
            (x['pnl']/x['ret'])*x['bm'] if x['ret']!=0 else x['hk_stock_pre_amount'] * x['bm'] / 10000, axis=1)
        # stats['bm_pnl'] = (stats['pnl']/stats['ret']) * stats['bm']
        # stats['bm_pnl'] = stats['hk_stock_pre_amount'] * stats['bm'] / 10000
    return stats


def cal_type_1(stats):
    """
    cal type 1:
    normal model alpha
    """
    # summary in stats, insert into DB
    # print(stats['strategy_id'],stats['stock_pre_amount']+stats['hk_stock_pre_amount'],stats['stock_pre_mv'])
    # 分母按占用资金算，考虑头寸为负的情况
    stats['ret'] = stats.apply(lambda x: \
        x['pnl']/(x['stock_pre_mv']+max(0,x['trade_net']))*10000 \
        if x['stock_pre_mv']+x['stock_mv']>0 \
            else (x['pnl']/(x['stock_pre_mv']+min(0,x['trade_net']))*10000 \
            if x['stock_pre_mv']+x['stock_mv']<0 \
            else 0), axis=1)
    # 头寸为负时，乘以负号
    stats['alpha'] = stats.apply(lambda x: x['ret'] - x['bm'] 
        if x['stock_pre_mv']>0 or x['stock_mv']>0 
        else (x['bm'] - x['ret'] if x['stock_pre_mv']<0 or x['stock_mv']<0 
              else 0), axis=1)
    # pos_ret和trade_ret同ret
    stats['pos_ret'] = stats.apply(lambda x: \
        x['pos_pnl']/(x['stock_pre_mv']+max(0,x['trade_net']))*10000 
        if x['stock_pre_mv']+x['stock_mv']>0 
            else(x['pos_pnl']/(x['stock_pre_mv']+min(0,x['trade_net']))*10000 
            if x['stock_pre_mv']+x['stock_mv']<0
            else 0), axis=1)
    stats['trade_ret'] = stats.apply(lambda x: \
        x['trade_pnl']/(x['stock_pre_mv']+max(0,x['trade_net']))*10000 
        if x['stock_pre_mv']+x['stock_mv']>0 
            else(x['trade_pnl']/(x['stock_pre_mv']+min(0,x['trade_net']))*10000 
            if x['stock_pre_mv']+x['stock_mv']<0
            else 0), axis=1)
    # 假设ret=0的情况是既无持仓也无交易
    stats['bm_pnl'] = stats.apply(lambda x: \
        (x['pnl']/x['ret'])*x['bm'] if x['ret']!=0 else x['stock_pre_mv'] * x['bm'] / 10000, axis=1)
    # stats['bm_pnl'] = (stats['stock_pre_amount']+stats['hk_stock_pre_amount']) * stats['bm'] / 10000
    return stats

def daily_statistics(date_ = calendar.get_trading_date(), acc_lists = None, new_account = None,
    to_db = True):
    '''
    Parameters
    ----------
    date_ : str, yyyymmdd
        DESCRIPTION. The default is calendar.get_trading_date().
    acc_lists : str, key word contained in strategy_id
        DESCRIPTION. The default is None.
    new_account : dict
        DESCRIPTION. The default is None.
    to_db : True or False
        DESCRIPTION. The default is True.
    '''
    # data pre-process    
    y_date = calendar.get_trading_date(date_, -1)        
    pos = get_pos(y_date)
    trade = get_trade(date_)
    pos,trade = get_jbm_500(pos,trade)
    if acc_lists is None:
        acc_lists = ALLOCATION['strategy_id'].drop_duplicates().tolist()
    else:
        pos = pos[pos['strategy_id'].str.contains(acc_lists)].reset_index(drop=True)
        trade = trade[trade['strategy_id'].str.contains(acc_lists)].reset_index(drop=True)
        
    BM = get_benchmark_info()
    stats = pd.DataFrame()
    
    if not pos.empty:
        # deal with diviend
        pos_dvd, pos =  get_dividend_amount_quantity(date_, acc_lists, pos)

    #if there is no T-1 pos and T trade, return [empty dataframe]
    if pos.empty and trade.empty:
        return stats
    
    # if calendar.is_trading_date(date_, exchange_='hk'):
    hkstock_prices = get_daily_quote(date_, 'hkstock', encoding_='gbk') 
    hkindex_prices = get_daily_quote(date_, 'hkindex')

    if calendar.is_trading_date(date_):
        stock_prices = get_daily_quote(date_, 'stock', encoding_='gbk') 
        bond_prices = get_daily_quote(date_, 'bond', encoding_='gbk')   # 添加债券行情
        future_prices = get_daily_quote(date_, 'future')
        index_prices = get_daily_quote(date_, 'index')
        fund_prices = get_daily_quote(date_, 'fund', encoding_='gbk')    
        option_prices = get_daily_quote(date_, 'option', encoding_='gbk') 
        forex_prices = get_daily_quote(date_, 'forex')
        cta_prices = get_daily_quote(date_, 'cta')
        cta_prices['pre_settle'] = cta_prices['close']-cta_prices['change_price']
        sc_member = get_sc_members(calendar.get_trading_date(date_, offset_=-1))
        # if calendar.is_trading_date(date_, exchange_='hk'):
        index_prices = index_prices.append(hkindex_prices, ignore_index=True)
        tmp_fu_price = future_prices.loc[:,['symbol','pre_close','close']]
        tmp_fu_price['change_price'] = tmp_fu_price['close'] - tmp_fu_price['pre_close']
        tmp_fu_price['change_rate'] = (tmp_fu_price['close'] / tmp_fu_price['pre_close'] - 1) * 100
        index_prices = index_prices.append(tmp_fu_price)
    else:
        if calendar.is_trading_date(date_, exchange_='hk'):
            index_prices = hkindex_prices
            forex_prices = get_daily_quote(calendar.get_trading_date(date_), 'forex')
    
    BM = BM.merge(index_prices, left_on='bm_symbol', right_on='symbol', how='left') 
    account_detail = get_account_detail(calendar.get_trading_date(date_, -1), new_account)
    
    def cal_pos_stock(pos):
        if not pos.empty:
            pos = insert_market_data(pos, stock_prices)
            ret = cal_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'stock_pos_pnl', 'stock_amount', 'stock_pre_amount', 'marginfee', 'stock_pos_counts'])
        return ret

    def cal_trade_stock(trade):
        if not trade.empty:
            trade = insert_market_data(trade, stock_prices)
            ret = cal_trade_return(trade, sc_member)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'stock_trade_pnl', 'stock_fee', 'stock_commission', 'stock_trade_net_close', 'stock_buy', 'stock_sell'])
        return ret

    def cal_pos_bond(pos):
        if not pos.empty:
            pos = insert_market_data(pos, bond_prices)
            ret = cal_bond_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'bond_pos_pnl', 'bond_amount', 'bond_pre_amount', 'bond_pos_counts'])
        return ret

    def cal_trade_bond(trade):
        if not trade.empty:
            trade = insert_market_data(trade, bond_prices)
            ret = cal_bond_trade_return(trade)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'bond_trade_pnl', 'bond_fee', 'bond_commission', 'bond_trade_net_close', 'bond_buy', 'bond_sell'])
        return ret

    def cal_pos_future(pos):
        if not pos.empty:
            pos = pos.merge(future_prices[['symbol','settle','pre_settle']], on='symbol', how='left')
            ret = cal_future_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id','future_pos_pnl','future_amount','future_pre_amount'])
        return ret

    def cal_trade_future(trade):
        if not trade.empty:
            trade = trade.merge(future_prices[['symbol','settle']], on='symbol', how='left')        
            ret = cal_future_trade_return(trade)
        else:
            ret = pd.DataFrame(columns=['strategy_id','future_trade_pnl','future_commission','future_trade_net_close','future_buy','future_sell'])
        return ret

    def cal_pos_fund(pos):
        if not pos.empty:
            pos = insert_market_data(pos, fund_prices)
            ret = cal_fund_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id','fund_pos_pnl','fund_amount'])
        return ret

    def cal_trade_fund(trade):
        if not trade.empty:
            trade = insert_market_data(trade, fund_prices)
            ret = cal_fund_trade_return(trade)
        else:
            ret = pd.DataFrame(columns=['strategy_id','fund_trade_pnl','fund_trade_net_close','fund_buy','fund_sell','fund_commission'])
        return ret

    def cal_pos_hk(pos):
        if not pos.empty:
            pos = insert_market_data(pos, hkstock_prices)
            ret = cal_hk_pos_return(pos, forex_prices.loc[forex_prices['symbol'].str.contains('HKDCNY'), 'close'].to_numpy()[0])
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'hk_stock_pos_pnl', 'hk_stock_amount', 'hk_stock_pre_amount', 'hk_stock_pos_counts', 'hk_stock_forex'])
        return ret

    def cal_trade_hk(trade):
        if not trade.empty:
            trade = insert_market_data(trade, hkstock_prices)
            ret = cal_hk_trade_return(trade, forex_prices.loc[forex_prices['symbol'].str.contains('HKDCNY'), 'close'].to_numpy()[0])
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'hk_stock_trade_pnl', 'hk_stock_fee', 'hk_stock_commission', 
                                'hk_stock_trade_net_close', 'hk_stock_buy', 'hk_stock_sell'])
        return ret
    
    def cal_pos_option(pos):
        if not pos.empty:
            pos = pos.merge(option_prices[['symbol','settle','change_price','multiplier']], on='symbol', how='left')          
            ret = cal_option_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'option_pos_pnl', 'option_amount'])
        return ret

    def cal_trade_option(trade):
        if not trade.empty:
            trade = trade.merge(option_prices[['symbol','settle','multiplier']], on='symbol', how='left')        
            ret = cal_option_trade_return(trade, date_)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'option_trade_pnl', 'option_commission', 'option_buy', 'option_sell', 'option_trade_net_close'])
        return ret

    def cal_pos_cta(pos):
        if not pos.empty:
            pos = pos.merge(cta_prices[['symbol','pre_settle','settle','multiplier']], on='symbol', how='left')          
            ret = cal_cta_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'cta_pos_pnl', 'cta_amount'])
        return ret

    def cal_trade_cta(trade):
        if not trade.empty:
            trade = trade.merge(cta_prices[['symbol','settle','multiplier']], on='symbol', how='left')        
            ret = cal_cta_trade_return(trade, date_)
        else:
            ret = pd.DataFrame(columns=['strategy_id', 'cta_trade_pnl', 'cta_commission', 'cta_buy', 'cta_sell', 'cta_trade_net_close'])
        return ret

    # calculation stock pnl/alpha
    pos_stats, trade_stats = pd.DataFrame(columns=['strategy_id']), pd.DataFrame(columns=['strategy_id'])

    sec_type_list = ['STOCK','BOND','FUTURE','OPTION','FUND','HK','CTA']
    # if not calendar.is_trading_date(date_, exchange_='hk'):
    #     sec_type_list = ['STOCK','FUTURE','OPTION','FUND']
    # elif not calendar.is_trading_date(date_):
    #     sec_type_list = ['HK']
    
    for sec_type in sec_type_list:
        if not pos.empty:      
            _pos = pos[pos['security_type']==sec_type]    
        else:
            _pos = pd.DataFrame()
        _f = 'cal_pos_' + sec_type.lower()
        _pos_stats = eval(_f)(_pos)
        
        if not trade.empty:
            _trade = trade[trade['security_type']==sec_type]
        else:   
            _trade = pd.DataFrame()
        _f = 'cal_trade_' + sec_type.lower()
        _trade_stats = eval(_f)(_trade)

        pos_stats = pos_stats.merge(_pos_stats, how='outer', on=['strategy_id'])
        trade_stats = trade_stats.merge(_trade_stats, how='outer', on=['strategy_id'])
    
    pos_stats = pos_stats.merge(pos_dvd, on='strategy_id', how='outer').fillna(0)
    stats = merge_pos_trade_stats(date_, pos_stats, trade_stats, account_detail, BM)

    #### 选择不同cal_type ####
    cal_types = ALLOCATION['cal_type'].unique().tolist()
    for t in cal_types:
        strategy_ids = ALLOCATION.loc[ALLOCATION['cal_type']==t, 'strategy_id'].tolist()
        tmp_stats = stats[stats['strategy_id'].isin(strategy_ids)]
        if not tmp_stats.empty:
            log.info(f'Deal Cal Type {t}')
            if t=='1':
                args = [tmp_stats]
                kwargs = {}
            elif t=='2':
                args = [tmp_stats]
                kwargs = {}
            elif t=='3':
                args = [tmp_stats]
                kwargs = {}

            tmp_stats = eval(f'cal_type_{t}')(*args, **kwargs)

            performance_stats = tmp_stats[['trade_dt','strategy_id','ret','bm','alpha',
            'pos_ret','trade_ret','pnl','pos_pnl','trade_pnl','stock_mv','stock_pre_mv',
            'fee','commission','buy','sell','trade_net','stock_mv_ratio',
            'product_asset','stock_turnover','update_time','bm_pnl','dvd_amount','strategy_name','marginfee']]
            if to_db and (not performance_stats.empty):
                save_result(performance_stats, attributiondb, '"public"."performance_1"', ['trade_dt', 'strategy_id'])

            detail_stats = tmp_stats[['trade_dt','strategy_id','stock_pos_pnl','stock_trade_pnl',
            'stock_buy','stock_sell','stock_turnover','stock_fee','stock_commission','stock_amount','stock_pos_counts',
            'future_pos_pnl','future_trade_pnl','future_buy','future_sell','future_commission','future_amount',
            'option_pos_pnl','option_trade_pnl','option_commission',
            'fund_pos_pnl','fund_trade_pnl','fund_buy','fund_sell','fund_commission','fund_amount',
            'hk_stock_pos_pnl','hk_stock_trade_pnl','hk_stock_buy','hk_stock_sell','hk_stock_turnover',
            'hk_stock_fee','hk_stock_commission','hk_stock_amount','hk_stock_pos_counts','hk_stock_forex',
            'cta_pos_pnl','cta_trade_pnl','cta_buy','cta_sell','cta_commission','cta_amount','marginfee']]
            if to_db and (not detail_stats.empty):
                save_result(detail_stats, attributiondb, '"public"."classcification_detail"', ['trade_dt', 'strategy_id'])

    # calculation dividend of T
    pos_end = get_pos(date_)
    if not pos_end.empty:
        r = calculate_dvd(date_, pos_end)
        if not r.empty:
            attributiondb.upsert('dividend_detail', df_=r, keys_=['ex_dt','strategy_id','symbol'])

    strategy_ids_ = ['12_ZS', '82_ZS', '101_ZS', '102B_ZS', 
                     '80F_ZS', '102C_ZS', '101C_ZSXK', '102C_ZSXK', 
                     '99A_FANCTA100', '99B_FANCTA100', '100A_FANCTA100', '104A_FANCTA100', 
                     '101A_CO', '80D_FANCTA100', '93I_FANCTA100']
    bm_pnls_ = [30*0.015/245*10000, 30*0.015/245*10000, 60*0.015/245*10000, 45*0.015/245*10000,
                60*0.015/245*10000, 600*0.015/245*10000, 45*0.015/245*10000, 360*0.015/245*10000,
                300*0.015/245*10000, 300*0.015/245*10000, 500*0.015/245*10000, 900*0.015/245*10000,
                500*0.015/245*10000, 900*0.015/245*10000, 300*0.015/245*10000]
    cal_special_bm_pnl(date_, strategy_ids_, bm_pnls_)
    return stats

def cal_special_bm_pnl(date_, strategy_ids_, bm_pnls_):
    # add bm pnl to non stock strategy (future, option)
    template = '''
        update "performance_1" set bm_pnl={} where strategy_id = '{}' and trade_dt='{}';
    '''
    sql = ''
    for i in range(len(strategy_ids_)):
        sql = sql + template.format(bm_pnls_[i], strategy_ids_[i], date_)
    attributiondb.execute(sql)


# =============================================================================
# 增加对performance的修改
# 作者：JL
# =============================================================================
#### 筛选阈值 ####
asset_ratio = 0.005
stock_ratio = 0.05
net_ratio = 0.5
adj_fee_rate = 0.001
weight_ratio = 2/3
test_strategy = ['CO','PA','KZZ']
#################

def new_cal_bm_vwap(date_):
    #### 计算bm_vwap ####
    # 读取date数据
    sql = f'''
            select trade_dt 
            from JasperCalendar
            where trade_dt = '{date_}'
            order by trade_dt
          '''
    date = pd.read_sql_query(sql, jasper_nav_reader)
    date = pd.Series(date['trade_dt']).apply(str)
    # 读取135服务器上的行情数据
    vwap_data = pd.DataFrame(columns=['date','sse50_buy','hs300_buy','csi500_buy','csi1000_buy',
                                      'sse50_sell','hs300_sell','csi500_sell','csi1000_sell'])
    vwap_data['date'] = date.values
    for i in range(len(date)):
        #读取135上的行情数据,bm,prices
        f_csv = pd.read_csv('\\\\10.144.64.135\\prod\\benchmark\\'+date[i]+'.bm.csv', encoding='gbk')
        temp_data = pd.DataFrame(f_csv)
        bm_data = temp_data[['date','symbol','sse50','hs300','csi500','csi1000']]
        f_csv = pd.read_csv('\\\\10.144.64.135\\prod\\prices\\'+date[i]+'.prices.csv', encoding='gbk')
        temp_data = pd.DataFrame(f_csv)
        price_data = temp_data[['symbol','date','prevclose','close','vwap']]
        price_data['vwap_rate_buy'] = (price_data['close']/price_data['vwap']-1)*10000
        price_data['vwap_rate_sell'] = (price_data['vwap']/price_data['prevclose']-1)*10000
        #合并，计算加权平均vwap
        temp_data = pd.merge(bm_data, price_data, how='left', on=['date','symbol'])
        vwap_data.sse50_buy.iloc[i] = np.sum(temp_data['sse50'] * temp_data['vwap_rate_buy'])/np.sum(temp_data['sse50'])
        vwap_data.hs300_buy.iloc[i] = np.sum(temp_data['hs300'] * temp_data['vwap_rate_buy'])/np.sum(temp_data['hs300'])
        vwap_data.csi500_buy.iloc[i] = np.sum(temp_data['csi500'] * temp_data['vwap_rate_buy'])/np.sum(temp_data['csi500'])
        vwap_data.csi1000_buy.iloc[i] = np.sum(temp_data['csi1000'] * temp_data['vwap_rate_buy'])/np.sum(temp_data['csi1000'])
        vwap_data.sse50_sell.iloc[i] = np.sum(temp_data['sse50'] * temp_data['vwap_rate_sell'])/np.sum(temp_data['sse50'])
        vwap_data.hs300_sell.iloc[i] = np.sum(temp_data['hs300'] * temp_data['vwap_rate_sell'])/np.sum(temp_data['hs300'])
        vwap_data.csi500_sell.iloc[i] = np.sum(temp_data['csi500'] * temp_data['vwap_rate_sell'])/np.sum(temp_data['csi500'])
        vwap_data.csi1000_sell.iloc[i] = np.sum(temp_data['csi1000'] * temp_data['vwap_rate_sell'])/np.sum(temp_data['csi1000'])
    vwap_data.columns = ['date','#1_buy','#2_buy','#3_buy','#4_buy','#1_sell','#2_sell','#3_sell','#4_sell']
    return vwap_data

def new_read_performance(date_):
    #### 读取performance_1和benchmark ####
    sql = f'''
            select trade_dt, a.strategy_id, ret, a.bm, alpha, pos_ret, trade_ret, pnl, pos_pnl, trade_pnl, 
            stock_mv, stock_pre_mv, fee, commission, buy, sell, trade_net, stock_mv_ratio, product_asset, 
            stock_turnover, a.update_time, bm_pnl, dvd_amount, a.strategy_name, marginfee, 
            strategy.bm as bm_formula, strategy.cal_type, adj_type, product_id, allocation.parents 
            from "performance_1" a
            left join strategy on strategy.strategy_name = a.strategy_name 
            left join allocation on allocation.strategy_id = a.strategy_id
            where trade_dt = '{date_}'
            order by trade_dt
          '''
    performance_new = pd.read_sql_query(sql, attr_reader)
    return performance_new

def new_cal_adj_fee(date_,performance_new):
    #### 按交易日和账户计算和分配adj_fee ####
    performance_new['adj_fee'] = 0
    date_list = np.unique(performance_new.trade_dt)
    for i in date_list:
        date = i
        temp_per = performance_new[(performance_new.trade_dt==date) & (performance_new.adj_type!='0')]
        product = np.unique(temp_per.product_id)
        for j in product:
            if not temp_per[(temp_per.product_id==j) & (temp_per.adj_type=='1')].empty and \
                not temp_per[(temp_per.product_id==j) & (temp_per.adj_type=='-1')].empty:
                temp_data = temp_per[temp_per.product_id==j]
                temp_data.adj_fee[temp_data.adj_type=='-1'] = \
                    abs(temp_data.trade_net[temp_data.adj_type=='-1'])*adj_fee_rate*(-1)
                tot_adj_fee = abs(np.sum(temp_data.adj_fee[temp_data.adj_type=='-1']))
                weight = temp_data.stock_pre_mv[temp_data.adj_type=='1']
                temp_data.adj_fee[temp_data.adj_type=='1'] = tot_adj_fee*weight/np.sum(weight)
                temp_per.adj_fee[temp_per.product_id==j] = temp_data.adj_fee.fillna(0)
        performance_new.adj_fee[(performance_new.trade_dt==date) & (performance_new.adj_type!='0')] = \
            temp_per.adj_fee
    return performance_new

def new_cal_alpha(date_,performance_new,vwap_data):
    log.info(f'Calculation new alpha of {date_}')
    #### 计算trade_bm_rate和trade_bm_pnl，添加到performance_new ####
    bm_id = ['#1','#2','#3','#4']
    performance_new['trade_bm_rate_buy'] = 0
    performance_new['trade_bm_pnl_buy'] = 0
    performance_new['trade_bm_rate_sell'] = 0
    performance_new['trade_bm_pnl_sell'] = 0
    # performance_new['alpha_new'] = 0
    # performance_new['bm_pnl_new'] = 0
    performance_new['special'] = 0
    performance_new['weight'] = 0
    for i in tqdm(range(performance_new.shape[0])):
        temp_strategy = performance_new.strategy_name.iloc[i]
        temp_date = performance_new.trade_dt.iloc[i]
        temp_bm_f = performance_new.bm_formula.iloc[i]
        temp_bm_f = temp_bm_f.replace('#5','#2')        # HSHKI --> h00300.CSI
        temp_bm_f = temp_bm_f.replace('#6','#3')        # IC00 --> h00905.CSI
        temp_bm_rate = vwap_data[vwap_data['date']==temp_date]
        # 计算bm_trade_rate
        if '#1' in temp_bm_f or '#2' in temp_bm_f or '#3' in temp_bm_f or '#4' in temp_bm_f:
            temp_buy_rate = zip(bm_id, temp_bm_rate.iloc[:,1:5].values.tolist()[0])
            temp_sell_rate = zip(bm_id, temp_bm_rate.iloc[:,5:9].values.tolist()[0])
            temp_bm_buy,temp_bm_sell = temp_bm_f,temp_bm_f
            for x,y in temp_buy_rate:
                temp_bm_buy = temp_bm_buy.replace(x,str(y))
            temp_bm_buy = eval(temp_bm_buy)
            for x,y in temp_sell_rate:
                temp_bm_sell = temp_bm_sell.replace(x,str(y))
            temp_bm_sell = eval(temp_bm_sell)
            # 保存pnl、买入量和卖出量
            pnl = performance_new.pnl.iloc[i]
            buy_amount = performance_new.buy.iloc[i]
            sell_amount = performance_new.sell.iloc[i]
            adj_fee = performance_new.adj_fee.iloc[i]
            # 按正负头寸区分rate
            if performance_new.stock_pre_mv.iloc[i]+performance_new.stock_mv.iloc[i]>0:    # 正头寸
                performance_new.trade_bm_rate_buy.iloc[i] = round(temp_bm_buy, 4)
                performance_new.trade_bm_rate_sell.iloc[i] = round(temp_bm_sell, 4)
                trade_bm_pnl_buy = performance_new.buy.iloc[i]*temp_bm_buy/10000
                trade_bm_pnl_sell = performance_new.sell.iloc[i]*temp_bm_sell/10000
                bm = performance_new.bm.iloc[i]/10000
                pos_amount = max(0,performance_new.stock_pre_mv.iloc[i]+performance_new.sell.iloc[i])
                # amount = performance_new.stock_pre_mv.iloc[i]+max(0,buy_amount+sell_amount)
                amount = weight_ratio*performance_new.stock_pre_mv.iloc[i]+\
                         (1-weight_ratio)*performance_new.stock_mv.iloc[i]
            elif performance_new.stock_pre_mv.iloc[i]+performance_new.stock_mv.iloc[i]<0:    # 负头寸
                # 假设所有卖出都是开仓，所有买入都是平仓
                performance_new.trade_bm_rate_buy.iloc[i] = round(temp_bm_sell, 4)
                performance_new.trade_bm_rate_sell.iloc[i] = round(temp_bm_buy, 4)
                trade_bm_pnl_buy = performance_new.sell.iloc[i]*temp_bm_buy/10000
                trade_bm_pnl_sell = performance_new.buy.iloc[i]*temp_bm_sell/10000
                bm = -performance_new.bm.iloc[i]/10000
                pos_amount = abs(min(0,performance_new.stock_pre_mv.iloc[i]+performance_new.buy.iloc[i]))
                # amount = abs(performance_new.stock_pre_mv.iloc[i]+min(0,buy_amount+sell_amount))
                amount = weight_ratio*abs(performance_new.stock_pre_mv.iloc[i])+\
                         (1-weight_ratio)*abs(performance_new.stock_mv.iloc[i])
            else:   # 昨持仓+今持仓=0
                print('昨持仓+今持仓==0:', i)
                performance_new.trade_bm_rate_buy.iloc[i] = 0
                performance_new.trade_bm_rate_sell.iloc[i] = 0
                trade_bm_pnl_buy = 0
                trade_bm_pnl_sell = 0
                bm = 0
                pos_amount = 0
                amount = abs(buy_amount+sell_amount)
            # buy和sell视为open和close
            performance_new.trade_bm_pnl_buy.iloc[i] = round(trade_bm_pnl_buy, 4)
            performance_new.trade_bm_pnl_sell.iloc[i] = round(trade_bm_pnl_sell, 4)
            performance_new.bm_pnl.iloc[i] = round(bm * pos_amount, 4)
            alpha_new = (pnl - bm*pos_amount - trade_bm_pnl_buy + trade_bm_pnl_sell + adj_fee)/amount*10000 if amount!=0 else 0
            performance_new.alpha.iloc[i] = round(alpha_new, 4)
            performance_new.weight.iloc[i] = round(amount, 4)
        else:
            performance_new.trade_bm_pnl_buy.iloc[i] = 0
            performance_new.trade_bm_pnl_sell.iloc[i] = 0
            performance_new.trade_bm_rate_buy.iloc[i] = 0
            performance_new.trade_bm_rate_sell.iloc[i] = 0
            performance_new.alpha.iloc[i] = performance_new.alpha.iloc[i]
            performance_new.bm_pnl.iloc[i] = performance_new.bm_pnl.iloc[i]
            if performance_new.ret.iloc[i]!=0:
                performance_new.weight.iloc[i] = abs(performance_new.pnl.iloc[i]/performance_new.ret.iloc[i])*10000
            else:
                performance_new.weight.iloc[i] = performance_new.product_asset.iloc[i]
        '''
        标记special：
        1、asset_ratio: 持仓规模低于产品资产的0.5%
        2、stock_ratio: 持仓规模低于前20个交易日平均持仓的5%
        3、net_ratio: 减仓超过50%，加仓超过100%
        4、测试策略special均为0
        '''
        if performance_new.cal_type.iloc[i]!='3':      #不等于3即为股票类策略
            temp_id = performance_new.strategy_id.iloc[i]
            temp_strategy = performance_new.strategy_name.iloc[i]
            temp_data = performance_new[performance_new['strategy_id']==temp_id]
            temp_stock_mv = temp_data.stock_mv[(temp_data['trade_dt']<temp_date) & (temp_data['special']==0)]
            trade_net = performance_new.trade_net.iloc[i]
            stock_pre_mv = performance_new.stock_pre_mv.iloc[i]
            # 判断是否是测试策略
            if temp_strategy not in test_strategy:
                # 第一个交易日，加仓or测试
                if temp_stock_mv.empty:
                    if abs(performance_new.stock_mv.iloc[i]) < asset_ratio * performance_new.product_asset.iloc[i]:
                        performance_new.special.iloc[i] = 1
                    else:
                        performance_new.special.iloc[i] = 0
                    # performance_new.special.iloc[i] = 4
                # 非第一个交易日
                else:
                    # 判断asset_ratio
                    if abs(performance_new.stock_pre_mv.iloc[i]) < asset_ratio * performance_new.product_asset.iloc[i] and \
                        abs(performance_new.stock_mv.iloc[i]) < asset_ratio * performance_new.product_asset.iloc[i]:
                        performance_new.special.iloc[i] = 1
                    # 判断stock_ratio
                    elif abs(performance_new.stock_pre_mv.iloc[i]) < stock_ratio * np.mean(temp_stock_mv[-20:]) and \
                        abs(performance_new.stock_mv.iloc[i]) < stock_ratio * np.mean(temp_stock_mv[-20:]):
                        performance_new.special.iloc[i] = 2
                    # 判断net_ratio
                    elif (trade_net>0 and stock_pre_mv/(stock_pre_mv+trade_net)<net_ratio) \
                        or (trade_net<0 and -trade_net/stock_pre_mv>net_ratio):
                        performance_new.special.iloc[i] = 3
                    else:
                        performance_new.special.iloc[i] = 0
        else:
            performance_new.special.iloc[i] = 0      #期货类策略不标记
    # 对列重新排序
    col = performance_new.columns.drop(['adj_type','product_id'])
    # col = col.drop('bm_pnl_new').insert(22,'bm_pnl_new')
    # col = col.drop('alpha_new').insert(5,'alpha_new')
    performance_new = performance_new[col]

    #### 数据库操作 ####
    # 删除已存在记录
    sql = f'''
            delete from performance_1 
            where trade_dt = '{date_}'
          '''
    attr_reader.execute(sql)
    # 保存数据到数据库
    performance_new.drop_duplicates().to_sql('performance_1',attr_reader,index=False,if_exists='append')
    # 删除无效记录
    sql = f'''
            delete from performance_1 
            where pnl=0 and stock_mv=stock_pre_mv and 
            trade_dt = '{date_}'
          '''
    attr_reader.execute(sql)


def new_cal_fancta(date_):
    log.info(f'Correct FANCTA100 of {date_}')
    #### 修正数据FANCTA100,ret,alpha,weight ####
    sql = f'''
          select trade_dt,strategy_id,ret,alpha,pnl,product_asset 
          from performance_1 
          where strategy_name = 'FANCTA100' 
          and trade_dt = '{date_}' 
          order by trade_dt
          '''
    cta_data = pd.read_sql_query(sql, attr_reader)    
    template = '''
        update performance_1 set ret={},alpha={},weight={} 
        where strategy_id = '{}' and trade_dt='{}';
    '''
    sql = ''
    for i in tqdm(range(cta_data.shape[0])):
        strategy_id = cta_data.strategy_id.iloc[i]
        trade_dt = cta_data.trade_dt.iloc[i]
        asset = cta_data.product_asset.iloc[i]
        pnl = cta_data.pnl.iloc[i]
        if '104' in strategy_id:
            if trade_dt <= '20200821':
                ret = pnl/asset*2*10000
                alpha = pnl/asset*2*10000
                weight = asset/2
            else:
                ret = pnl/asset*10000
                alpha = pnl/asset*10000
                weight = asset
        elif '80' in strategy_id:
            if trade_dt <= '20191217':
                ret = pnl/5000000*10000
                alpha = pnl/5000000*10000
                weight = 5000000
            elif trade_dt <= '20201216':
                ret = pnl/9000000*10000
                alpha = pnl/9000000*10000
                weight = 9000000
            else:
                ret = pnl/1400000*10000
                alpha = pnl/1400000*10000
                weight = 1400000
        elif '99' in strategy_id:
            if trade_dt <= '20200313':
                ret = pnl/asset*3*10000
                alpha = pnl/asset*3*10000
                weight = asset/3
            elif trade_dt <= '20200821':
                ret = pnl/asset*2*10000
                alpha = pnl/asset*2*10000
                weight = asset/2
            else:
                ret = pnl/asset*2*10000
                alpha = pnl/asset*2*10000
                weight = asset/2
        elif '93' in strategy_id:
            if trade_dt <= '20201217':
                ret = pnl/3000000*10000
                alpha = pnl/3000000*10000
                weight = 3000000
            elif trade_dt <= '20201221':
                ret = pnl/2000000*10000
                alpha = pnl/2000000*10000
                weight = 2000000
            else:
                ret = pnl/4000000*10000
                alpha = pnl/4000000*10000
                weight = 4000000
        elif '105' in strategy_id:
            ret = pnl/5000000*10000
            alpha = pnl/5000000*10000
            weight = 5000000
        else:   # 100
            if trade_dt <= '20200313':
                ret = pnl/asset*3*10000
                alpha = pnl/asset*3*10000
                weight = asset/3
            elif trade_dt <= '20200821':
                ret = pnl/asset*2*10000
                alpha = pnl/asset*2*10000
                weight = asset/2
            else:
                ret = pnl/asset*10000
                alpha = pnl/asset*10000
                weight = asset
        sql = sql + template.format(ret, alpha, weight, strategy_id, trade_dt)
        attr_reader.execute(sql)

def new_cal_arbt(date_):
    log.info(f'Correct ARBT of {date_}')
    #### 修正数据ARBT,ret,alpha,weight ####
    sql = f'''
          select trade_dt,strategy_id,ret,alpha,pnl,product_asset 
          from performance_1 
          where strategy_name = 'ARBT' 
          and trade_dt = '{date_}' 
          order by trade_dt
          '''
    arbt_data = pd.read_sql_query(sql, attr_reader)    
    template = '''
        update performance_1 set ret={},alpha={},weight={} 
        where strategy_id = '{}' and trade_dt='{}';
    '''
    sql = ''
    for i in tqdm(range(arbt_data.shape[0])):
        strategy_id = arbt_data.strategy_id.iloc[i]
        trade_dt = arbt_data.trade_dt.iloc[i]
        pnl = arbt_data.pnl.iloc[i]
        ret = arbt_data.ret.iloc[i]
        alpha = arbt_data.alpha.iloc[i]
        if '105B' in strategy_id:
            ret = pnl/40000000*10000
            alpha = ret
            weight = 40000000
        elif '101C' in strategy_id:
            ret = pnl/200000000*10000
            alpha = ret
            weight = 200000000
        else:
            weight = abs(pnl/ret)*10000
        sql = sql + template.format(ret, alpha, weight, strategy_id, trade_dt)
        attr_reader.execute(sql)


def new_cal_per_strategy(date_):
    log.info(f'Write performance_strategy of {date_}')
    #### 保存数据到performance_strategy ####
    # sqlstr = f'''
    #          select trade_dt,strategy_id,strategy_name,alpha,stock_mv,stock_pre_mv,weight 
    #          from performance_1 
    #          where strategy_name !='ZSINTRADAY' and strategy_name!='OTHER' and strategy_name!='DZ' 
    #           and trade_dt = '{date_}'
    #          order by trade_dt
    #          '''
    sqlstr = f'''
             select trade_dt,strategy_id,strategy_name,alpha,stock_mv,stock_pre_mv,weight from (
                 select trade_dt,strategy_id,strategy_name,alpha,stock_mv,stock_pre_mv,weight 
                 from performance_1 
                 where strategy_name!='ZSINTRADAY' and strategy_name!='OTHER' and strategy_name!='DZ' 
                 and trade_dt = '{date_}' and strategy_name != 'JBM'
                 union
                 select trade_dt,strategy_id,strategy_name,alpha,stock_mv,stock_pre_mv,weight 
                 from performance_1 
                 where trade_dt = '{date_}' and strategy_name = 'JBM' and special = 0) as foo
             order by trade_dt
             '''
    performance = pd.read_sql_query(sqlstr, attr_reader)
    f = lambda x: np.average(x.iloc[:,0], weights=x.iloc[:,1])
    per_alpha = performance.groupby(['trade_dt','strategy_name'])[['alpha','weight']].apply(f).reset_index()
    per_alpha = per_alpha.rename(columns={0:'alpha'})
    per_weight = performance.groupby(['trade_dt','strategy_name'])[['stock_mv','stock_pre_mv','weight']].sum().reset_index()
    per_strategy = per_alpha.merge(per_weight, how='left', on=['trade_dt','strategy_name'])
    per_strategy['test'] = 0
    per_strategy = per_strategy[['trade_dt','strategy_name','alpha','stock_mv','stock_pre_mv','weight','test']]

    # 删除已存在记录
    sql = f'''
            delete from performance_strategy 
            where trade_dt = '{date_}' and test = 0
          '''
    attr_reader.execute(sql)
    # 保存数据
    per_strategy.to_sql('performance_strategy',attr_reader,index=False,if_exists='append')


def new_cal_all(date_):
    vwap_data = new_cal_bm_vwap(date_)
    performance_new = new_read_performance(date_)
    performance_new = new_cal_adj_fee(date_, performance_new)
    new_cal_alpha(date_, performance_new, vwap_data)
    new_cal_fancta(date_)
    new_cal_arbt(date_)
    new_cal_per_strategy(date_)


if __name__ == "__main__":
    dt = calendar.get_trading_date(offset_=-1)
    # dt = '20201218'
    print(f'The date to calculate is \033[1;30;43m{dt}\033[0m, right or not? (1: right, others: wrong)')
    pyautogui.hotkey('ctrl','shift','i')
    ans = input()
    if ans=='1':
        print('You are calculating all strategy-performance:')
        step = ['1 --> check allocation',
                '2 --> calculate performance',
                '3 --> correct new part',
                '4 --> send test email',
                '5 --> send email to company']
        for i in range(len(step)):
            print('Please input the operation of the step:')
            print(f'{step[i]}, yes or no? (1: yes, 2: no and continue, other: no and break)')
            temp_ans = input()
            if temp_ans == '1':
                if i==0:       # 检查allocation配置
                    check_allocation(dt)
                elif i==1:     # 计算数据
                    daily_statistics(dt, to_db = True)
                elif i==2:     # 修正performance_1数据，保存performance_strategy
                    new_cal_all(dt)
                elif i==3:     # 发送测试邮件
                    at_night_report(dt, test=True)
                elif i==4:     # 发送群发邮件
                    at_night_report(dt, test=False)
                else:
                    print('Check the number of step!')
            elif temp_ans == '2':
                continue
            else:
                break
    else:
        print(f'The date \033[1;30;43m{dt}\033[0m is wrong!')

    # date_list = calendar.get_trading_calendar(from_='20200728', to_='20201219')
    # for d in date_list:
    #     print(f'Deal date \033[1;32;43m {d} \033[0m')
    #     daily_statistics(d, acc_lists='JBM500', to_db = True)
    #     new_cal_all(d)

    # check_allocation(d)
    
    # daily_statistics('20201023', to_db = True,
    #                 new_account=pd.DataFrame({'account_id':['113'], 'totalasset':[200000000]}))
    # check_allocation(tc.get_trading_date(offset_=-1)) # 检查allocation配置    
    # check_allocation('20201023') # 检查历史allocation配置
    # at_night_report(tc.get_trading_date(offset_=-1), test=True)
    # at_night_report(tc.get_trading_date(offset_=-1), test=False)

    # date_list = calendar.get_trading_calendar(from_='20190701', to_='20201210')
    # for d in date_list:
    #     print(f'Deal date {d}')
    #     check_allocation(d)
    pass
