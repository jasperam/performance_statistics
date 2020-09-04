import pandas as pd
import datetime

from ps.data_loader import get_pos, get_trade, get_benchmark_info, get_alloction, \
    get_account_detail, get_daily_quote, get_commission_rate, calculate_dvd, cal_option_volume, \
    get_sc_members, cal_cta_commission, get_dividend_amount_quantity
from ps.utils import save_result
from jt.utils.misc.log import Logger
from jt.utils.calendar.api_calendar import TradeCalendarDB
from jt.utils.db import PgSQLLoader
from jt.utils.misc import read_cfg
from jt.utils.time import datetime2string, num_of_week

FEE_RATE = 1e-3

MULTIPLIER_DICT = {
    'IC': 200,
    'IF': 300,
    'IH': 300
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
            return MULTIPLIER_DICT[str(x['symbol'])[0:2]]
        else:
            return 1 # stock/fund -> 1
    return _inner

def get_benchmark_rct(BM, date_):
    def _inner(strategy_):
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
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_margin, ['strategy_id','commission','min_commission']]
    pos = pos.merge(_allocation, on='strategy_id', how='left').fillna(0)
    pos['marginfee'] = pos.apply(lambda x: abs(x['stock_amount']) * (x['commission']-0.017) / 250 
                                if (x['commission']>0) and ('_RQ' in x['strategy_id']) else 0, 
                                axis=1) # Simple handling of currency interest rate of 1.7%
    pos['stock_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier'] - pos['marginfee']
    pos_stats = pos[['strategy_id', 'stock_pos_pnl', 'stock_amount', 'stock_pre_amount', 'marginfee']].groupby(
        by='strategy_id').sum().reset_index()
    tmp_counts = pos[['strategy_id','date']].groupby('strategy_id').count().reset_index()
    tmp_counts.rename(columns={'date':'stock_pos_counts'}, inplace=True)
    pos_stats = pos_stats.merge(tmp_counts, on='strategy_id', how='inner')
    return pos_stats

def cal_trade_return(trade, sc_member): 
    trade['multiplier'] = 1    
    trade['stock_trade_amount'] = abs(trade['volume'] * trade['price'] * trade['multiplier'])
    _allocation = ALLOCATION.loc[ALLOCATION.sec_type==_sec_type_stock, ['strategy_id','commission','min_commission']]
    trade = trade.merge(_allocation, on='strategy_id', how='left')    
    # ---------------------------------
    # deal with qfii_fee of prelude(96)
    qfii_fee_96 = ALLOCATION.loc[(ALLOCATION.sec_type=='QFII') & (ALLOCATION.product_id=='96'), 'commission'].to_numpy()[0]
    trade.loc[trade['strategy_id'].str.contains('96') & trade['symbol'].isin(sc_member['symbol']), 'commission'] = qfii_fee_96
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
    trade['future_commission'] = abs(trade['price']*trade['volume']*trade['multiplier'])*trade['commission']+trade['min_commission']
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
    pos['cta_pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier']
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
    stats['stock_amount'] = stats['stock_amount'] + stats['stock_trade_net_close']
    stats['hk_stock_amount'] = stats['hk_stock_amount'] + stats['hk_stock_trade_net_close']    
    stats['future_amount'] = stats['future_amount'] + stats['future_trade_net_close']
    stats['fund_amount'] = stats['fund_amount'] + stats['fund_trade_net_close']    
    stats['option_amount'] = stats['option_amount'] + stats['option_trade_net_close']
    stats['cta_amount'] = stats['cta_amount'] + stats['cta_trade_net_close']
    stats['stock_mv'] = stats['stock_amount'] + stats['hk_stock_amount']    
    stats['pos_pnl'] = stats['stock_pos_pnl'] + stats['dvd_amount'] + stats['future_pos_pnl'] + stats['fund_pos_pnl'] + stats['option_pos_pnl'] \
        + stats['hk_stock_pos_pnl'] + stats['cta_pos_pnl']
    stats['trade_pnl'] = stats['stock_trade_pnl'] + stats['future_trade_pnl'] + stats['fund_trade_pnl']  + stats['option_trade_pnl'] \
        + stats['hk_stock_trade_pnl'] + stats['cta_trade_pnl']
    stats['pnl'] = stats['pos_pnl'] + stats['trade_pnl']
    stats['buy'] = stats['stock_buy'] + stats['future_buy'] + stats['fund_buy'] + stats['option_buy'] + stats['hk_stock_buy'] + stats['cta_buy']
    stats['sell'] = stats['stock_sell'] + stats['future_sell'] + stats['fund_sell'] + stats['option_sell'] + stats['hk_stock_sell'] + stats['cta_sell']
    stats['trade_net'] = stats['buy'] + stats['sell']
    stats['commission'] = stats['stock_commission'] + stats['fund_commission'] + stats['hk_stock_commission'] + stats['option_commission'] + stats['future_commission'] \
        + stats['cta_commission']
    stats['fee'] = stats['stock_fee'] + stats['hk_stock_fee']
    stats['stock_turnover'] = stats.apply(lambda x: (x['stock_buy'] + abs(x['stock_sell'])) / x['stock_pre_amount']
        if x['stock_pre_amount'] > 0 else 0 , axis=1)
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
    model pnl based net asset
    """    
    # summary in stats, insert into DB   
    stats['ret'] = stats['pnl'] / stats['product_asset'] * 10000
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
            x['pnl']/x['hk_stock_pre_amount']*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)
        stats['alpha'] = stats['ret'] - stats['bm']    
        stats['pos_ret'] = stats.apply(lambda x: \
            x['pos_pnl']/x['hk_stock_pre_amount']*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)
        stats['trade_ret'] = stats.apply(lambda x: \
            x['trade_pnl']/x['hk_stock_pre_amount']*10000 if x['hk_stock_pre_amount']>0 else 0, axis=1)     
        stats['bm_pnl'] = stats['hk_stock_pre_amount'] * stats['bm'] / 10000

    return stats


def cal_type_1(stats):
    """
    cal type 1:
    normal model alpha
    """    
    # summary in stats, insert into DB  
    stats['ret'] = stats.apply(lambda x: \
        x['pnl']/(x['stock_pre_amount']+x['hk_stock_pre_amount'])*10000 
        if abs(x['stock_pre_amount']+x['hk_stock_pre_amount'])>0 else 0, axis=1)
    stats['alpha'] = stats.apply(lambda x: x['ret'] - x['bm'] 
        if (x['stock_pre_amount']+x['hk_stock_pre_amount'])>=0 else x['bm'] - x['ret'], axis=1)
    stats['pos_ret'] = stats.apply(lambda x: \
        x['pos_pnl']/(x['stock_pre_amount']+x['hk_stock_pre_amount'])*10000 
        if abs(x['stock_pre_amount']+x['hk_stock_pre_amount'])>0 else 0, axis=1)
    stats['trade_ret'] = stats.apply(lambda x: \
        x['trade_pnl']/(x['stock_pre_amount']+x['hk_stock_pre_amount'])*10000 
        if abs(x['stock_pre_amount']+x['hk_stock_pre_amount'])>0 else 0, axis=1)     
    stats['bm_pnl'] = (stats['stock_pre_amount']+stats['hk_stock_pre_amount']) * stats['bm'] / 10000
    
    return stats

def daily_statistics(date_=calendar.get_trading_date(), acc_lists=None, new_account=None,
    to_db = True):   
    # data pre-process    
    if acc_lists is None:
        acc_lists = ALLOCATION['strategy_id'].drop_duplicates().tolist()
    BM = get_benchmark_info()
    stats = pd.DataFrame()
    # if not calendar.is_trading_date(date_):
    #     tmp_date = calendar.get_trading_date(date_, -1)
    #     pos = get_pos(tmp_date, acc_lists)
    # else:
    y_date = calendar.get_trading_date(date_, -1)        
    pos = get_pos(y_date, acc_lists)
        
    if not pos.empty:
        # deal with diviend
        pos_dvd, pos =  get_dividend_amount_quantity(date_, acc_lists, pos)

    trade = get_trade(date_, acc_lists)

    #if there is no T-1 pos and T trade, return [empty dataframe]
    if pos.empty and trade.empty:
        return stats
    
    # if calendar.is_trading_date(date_, exchange_='hk'):
    hkstock_prices = get_daily_quote(date_, 'hkstock', encoding_='gbk') 
    hkindex_prices = get_daily_quote(date_, 'hkindex')

    if calendar.is_trading_date(date_):        
        stock_prices = get_daily_quote(date_, 'stock', encoding_='gbk') 
        future_prices = get_daily_quote(date_, 'future')
        index_prices = get_daily_quote(date_, 'index')
        fund_prices = get_daily_quote(date_, 'fund', encoding_='gbk')    
        option_prices = get_daily_quote(date_, 'option', encoding_='gbk') 
        forex_prices = get_daily_quote(date_, 'forex')
        cta_prices = get_daily_quote(date_, 'cta')
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

    def cal_pos_future(pos):
        if not pos.empty:
            pos = pos.merge(future_prices[['symbol','settle','pre_settle']], on='symbol', how='left')          
            ret = cal_future_pos_return(pos)
        else:
            ret = pd.DataFrame(columns=['strategy_id','future_pos_pnl','future_amount'])
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
            pos = pos.merge(cta_prices[['symbol','settle','change_price','multiplier']], on='symbol', how='left')          
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

    sec_type_list = ['STOCK','FUTURE','OPTION','FUND','HK','CTA'] 
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
            'pos_ret','trade_ret','pnl','pos_pnl','trade_pnl','stock_mv',
            'fee','commission','buy','sell','trade_net','stock_mv_ratio',
            'product_asset','stock_turnover','update_time','bm_pnl','dvd_amount','strategy_name','marginfee']]

            if to_db and (not performance_stats.empty):
                save_result(performance_stats, attributiondb, '"public"."performance"', ['trade_dt', 'strategy_id'])

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
    pos_end = get_pos(date_, acc_lists)  
    if not pos_end.empty:  
        r = calculate_dvd(date_, pos_end) 
        if not r.empty:
            attributiondb.upsert('dividend_detail', df_=r, keys_=['ex_dt','strategy_id','symbol'])

    strategy_ids_ = ['12_ZS','82_ZS', '101_ZS', '102B_ZS', '80F_ZS', '102C_ZS',
        '99A_FANCTA100','99B_FANCTA100','100A_FANCTA100', '104A_FANCTA100',
        '101C_ZSXK', '102C_ZSXK']
    bm_pnls_ = [30*0.015/245*10000, 30*0.015/245*10000, 60*0.015/245*10000, 45*0.015/245*10000, 60*0.015/245*10000, 300*0.015/245*10000, 
        239*0.015/245*10000, 352*0.015/245*10000, 540*0.03/245*10000, 1100*0.015/245*10000,
        45*0.015/245*10000 ,360*0.015/245*10000]
    cal_special_bm_pnl(date_, strategy_ids_, bm_pnls_)

    return stats

def cal_special_bm_pnl(date_, strategy_ids_, bm_pnls_):
    # add bm pnl to non stock strategy (future, option)
    template = '''
        update "performance" set bm_pnl={} where strategy_id = '{}' and trade_dt='{}';
    '''
    sql = ''
    for i in range(len(strategy_ids_)):
        sql = sql + template.format(bm_pnls_[i], strategy_ids_[i], date_)
    attributiondb.execute(sql)


if __name__ == "__main__":    
    # date_list = calendar.get_trading_calendar(from_='20200313', to_='20200317')
    # for d in date_list:
    #     print(f'Deal date {d}')
    #     acc_lists = ['01_HEDGE','06_HEDGE','07_HEDGE','13_HEDGE','52_HEDGE','55_HEDGE','55_JDHEDGE',
    #             '58_HEDGE','64A_HEDGE','79_HEDGE','79_JDHEDGE','80_HEDGE','80_ZSINTRADAY','82_HEDGE',
    #             '85B_HEDGE','93A_HEDGE','93A_JDHEDGE','93_HEDGE','93_JDHEDGE','06_OTHER','07_OTHER',
    #             '12_OTHER','13_OTHER','41_OTHER','52_OTHER','55_OTHER','58_OTHER','62_OTHER','64A_OTHER',
    #             '79_OTHER','80A_OTHER','80_OTHER','82_OTHER','84_OTHER','85A_OTHER','85B_OTHER','89_OTHER',
    #             '91_OTHER','93A_OTHER','93B_OTHER','93_OTHER','06_DV','01_DV','01_PETER','01_ZF502']
        # daily_statistics(d, ['99A_ARBT','99A_MJCTA']) #, new_account=pd.DataFrame({'account_id':['95'], 'totalasset':[2000000]})
    daily_statistics('20200901',['01_ZFM502','99A_FANCTA100','99B_FANCTA100','100A_FANCTA100', '104A_FANCTA100','102C_ZS'])
    # date_ = '20200103'
    # strategy_ids_ = ['100_FANCTA100']
    # bm_pnls_ = [133*0.03/245*10000]
    # cal_special_bm_pnl(date_,strategy_ids_,bm_pnls_)
    pass

