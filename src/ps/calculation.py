import pandas as pd
import datetime

from WindPy import w
from ps.data_loader import get_pos, get_trade, get_benchmark_info, get_alloction, \
    get_account_detail, get_index_wsd, get_index_wsi, get_price_dict, get_commission_rate, \
    insert_market_data, get_cal_type2_index_wsi
from jt.utils.misc.log import Logger
from jt.utils.calendar.api_calendar import TradeCalendarDB
from jt.utils.db import PgSQLLoader
from jt.utils.misc import read_cfg
from jt.utils.time import datetime2string

FEE_RATE = 1e-3
FU_FEE_RATE = 0.23e-4

MULTIPLIER_DICT = {
    'IC': 200,
    'IF': 300,
    'IH': 300
}

BENCHMARK = ['H00016.SH','H00300.CSI','H00905.CSI','H00852.SH']

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

dataloader = DataLoader()
calendar = TradeCalendarDB()
pgloader = PgSQLLoader()
pgloader.set_db_config(read_cfg('cfg/db.ini', package='ps')['attribution'])
log = Logger(module_name_=__name__)

ALLOCATION = get_alloction()
ALLOCATION = ALLOCATION.loc[ALLOCATION['strategy_id'].isin(['91_JD1000','55_ZF502'])]

def get_multiplier():
    def _inner(symbol):
        if str(symbol)[0:2] in MULTIPLIER_DICT.keys():
            return MULTIPLIER_DICT[str(symbol)[0:2]]
        else:
            return 1 # stock -> 1
    return _inner


def get_benchmark_rct(BM):
    def _inner(strategy_):
        f_ = ALLOCATION.loc[ALLOCATION['strategy_id']==strategy_, 'bm'].values[0]
        for x, y in zip(BM['bm_id'], BM['pct_chg'].astype(str)):
            f_ = f_.replace(x, y)
        return eval(f_)/100
    return _inner


def get_asset(account_detail):
    def _inner(strategy_):
        account_id = ALLOCATION.loc[ALLOCATION['strategy_id']==strategy_, 'product_id'].values[0]
        asset = account_detail.loc[account_detail['account_id']==account_id, 'totalasset'].values[0]
        return asset
    return _inner


def save_result(df_, table_name, keys_):    
    log.info(f'Save result to DB {table_name}')
    pgloader.upsert(table_name, df_=df_, keys_=keys_)    


def cal_pos_return(pos, prices_dict):    
    pos['multiplier'] = pos['symbol'].apply(get_multiplier())
    pos['pos_pnl'] = pos['change_price'] * pos['volume'] * pos['multiplier']
    pos['pos_amount'] = pos['close'] * pos['volume'] * pos['multiplier']
    pos['pos_pre_amount'] = pos['pre_close'] * pos['volume'] * pos['multiplier'] # cal ratio
    pos_stats = pos[['strategy_id', 'pos_pnl', 'pos_amount', 'pos_pre_amount']].groupby(
        by='strategy_id').sum().reset_index()
    return pos_stats


def cal_trade_return(trade, prices_dict):    
    trade['multiplier'] = trade['symbol'].apply(get_multiplier())
    trade['trade_amount'] = abs(trade['volume'] * trade['price'] * trade['multiplier'])
    trade['trade_net_closeprice'] = trade['volume'] * trade['close'] * trade['multiplier']
    trade['fee'] = trade.apply(lambda x:
        x['trade_amount'] * FEE_RATE if x['volume'] < 0 else 0, axis=1)      
    trade['commission'] = trade['trade_amount'] * trade['commission_rate']
    trade['trade_pnl'] = (trade['close']-trade['price']) * trade['volume'] - trade['fee'] - trade['commission']
    trade['buy'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']>0 else 0, axis=1)     
    trade['sell'] = trade.apply(lambda x: x['price'] * x['volume'] if x['volume']<0 else 0, axis=1)     
    trade_stats = trade[['strategy_id', 'trade_pnl', 'fee', 'commission', 'trade_net_closeprice', 'buy', 'sell']].groupby(
        by='strategy_id').sum().reset_index()
    return trade_stats


def merge_pos_trade_stats(date_, pos_stats, trade_stats, account_detail, BM):
    stats = pd.merge(pos_stats, trade_stats, how='outer', on='strategy_id', suffixes=('_pos','_trade')).fillna(0)
    stats['trade_dt'] = date_
    stats['bm'] = stats['strategy_id'].apply(get_benchmark_rct(BM))
    stats['bm'] = stats['bm'] * 10000
    stats['pnl'] = stats['pos_pnl'] + stats['trade_pnl']
    stats['mv'] = stats['pos_amount'] + stats['trade_net_closeprice']
    stats['trade_net'] = stats['buy'] + stats['sell']
    stats['turnover'] = stats.apply(lambda x:
        (x['buy'] + abs(x['sell'])) / x['pos_pre_amount']
        if x['pos_pre_amount'] > 0 else 0 , axis=1)
    stats['product_asset'] = stats['strategy_id'].apply(get_asset(account_detail)) 
    stats['mv_ratio'] = stats['mv'] / stats['product_asset']
    stats['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return stats


def cal_type_1(date_, stats):
    """
    cal type 1:
    normal model alpha
    """    
    # summary in stats, insert into DB   
    stats['ret'] = stats.apply(lambda x: \
        x['pnl']/x['pos_pre_amount']*10000 if x['pos_pre_amount']>0 else 0, axis=1)
    stats['alpha'] = stats['ret'] - stats['bm']    
    stats['pos_ret'] = stats.apply(lambda x: \
        x['pos_pnl']/x['pos_pre_amount']*10000 if x['pos_pre_amount']>0 else 0, axis=1)
    stats['trade_ret'] = stats.apply(lambda x: \
        x['trade_pnl']/x['pos_pre_amount']*10000 if x['pos_pre_amount']>0 else 0, axis=1)    
    
    return stats


def cal_type_2(date_, pos, trade, stats, prices_dict, w):
    """
    cal type 2:
    trade sensitive
    """  
    # deal with t0 sell
    def deal_t0_sell(trade_):
        ret = pd.DataFrame()
        if not trade_.empty:
            t0sell = trade_[(trade_.volume < 0) & (trade_.price > 0)] #T0 sell
            if not t0sell.empty:
                index_wsi = get_cal_type2_index_wsi(t0sell, w)
                for _sec in t0sell['symbol'].unique:
                # 1. 获取每笔成交的指数价格
                # 2. 根据成交金额计算每笔成交的比例
                # 3. 根据比例加权得到成交价和指数价格
                    _t = t0sell[t0sell['symbol']==_sec]
                    _t['trade_amount'] = abs(_t['volume'] * _t['price'] * _t['multiplier'])
                    _t['weight'] = _t['trade_amount'] / _t['trade_amount'].sum()
                    _t['000300.SH'] = _t['time'].apply(lambda x: index_wsi.loc[x, '000300.SH'])
                    _t['000905.SH'] = _t['time'].apply(lambda x: index_wsi.loc[x, '000905.SH'])
                    _t['000016.SH'] = _t['time'].apply(lambda x: index_wsi.loc[x, '000016.SH'])
                    _t['000852.SH'] = _t['time'].apply(lambda x: index_wsi.loc[x, '000852.SH'])
                    # date, symbol, volume, trade_amount, weight, price, 000300.SH...
                    ret.append()
                
                


    acc_id = pos.strategy_id.unique().tolist()
    y_trade = get_trade(y_date, acc_id)

    y_prices_dict = get_price_dict(date_, y_trade.symbol.unique()) 
    y_trade = insert_market_data(y_trade, y_prices_dict)
    

    for i in acc_id:
        df = pos[pos['strategy_id']==i]
        df['weight'] = df['pos_pre_amount']/df['pos_pre_amount'].sum()
        df['bm'] = stats.loc[stats['strategy_id']==i, 'bm'].to_numpy()[0]
        df['pos_alpha'] = df['change_rate'] - df['bm']
        

        df_sell, df_buy = cal_profit_rct()
    pass


def daily_statistics(date_):   
    # data pre-process    
    BM = get_benchmark_info()
    stats, symbol_stock, symbol_fu = pd.DataFrame(), pd.Series(), pd.Series()
    pos = get_pos(date_, ALLOCATION['strategy_id'].tolist())
    if not pos.empty:
        pos_stock = pos[pos['security_type']==_sec_type_stock]              
        symbol_stock = pos_stock['symbol']
        pos_fu = pos[pos['security_type']==_sec_type_future]
        symbol_fu = pos_fu['symbol']

    trade = get_trade(date_, ALLOCATION['strategy_id'].tolist())
    if not trade.empty:    
        trade_stock = trade[trade['security_type']==_sec_type_stock] 
            
        trade_stock['commission_rate'] = trade_stock['strategy_id'].apply(
            get_commission_rate(ALLOCATION))
        symbol_stock = symbol_stock.append(trade_stock['symbol'])
        trade_fu = trade[trade['security_type']==_sec_type_future]
        symbol_fu = symbol_fu.append(trade_fu['symbol'])

    #if there is no T-1 pos and T trade, return [empty dataframe]
    if pos.empty and trade.empty:
        return stats

    w.start()    
    factors = ["pct_chg","close","pre_close","chg"]
    index_stats_ = get_index_wsd(date_, BENCHMARK, factors, w)
    index_df_ = pd.DataFrame.from_dict(index_stats_).T
    BM = BM.merge(index_df_, left_on='bm_symbol', right_on='windcode', how='left').drop(['windcode','time'], axis=1)

    prices_dict = get_price_dict(date_, symbol_stock) 
    account_detail = get_account_detail(calendar.get_trading_date(date_, -1))
       
    # calculation stock pnl/alpha
    pos_stats, trade_stats = pd.DataFrame(), pd.DataFrame()
    if not pos_stock.empty:       
        pos_stock = insert_market_data(pos_stock, prices_dict)  
        pos_stats = cal_pos_return(pos_stock, prices_dict)
    else:
        pos_stats = pd.DataFrame(columns=['strategy_id', 'pos_pnl', 'pos_amount',  'pos_pre_amount'])
        
    if not trade_stock.empty:
        trade_stock = insert_market_data(trade_stock, prices_dict)
        trade_stats = cal_trade_return(trade_stock, prices_dict)
    else:
        trade_stats = pd.DataFrame(columns=['strategy_id', 'trade_pnl', 'fee', 'commission', 'trade_net_closeprice', 'buy', 'sell'])

    stats = merge_pos_trade_stats(date_, pos_stats, trade_stats, account_detail, BM)

    cal_types = ALLOCATION['cal_type'].unique().tolist()
    for t in cal_types:
        log.info(f'Deal Cal Type {t}')
        strategy_ids = ALLOCATION.loc[ALLOCATION['cal_type']==t, 'strategy_id'].tolist()
        stats = stats[stats['strategy_id'].isin(strategy_ids)]
        if t=='1':
            args = [date_, stats]
            kwargs = {}

        stats = eval(f'cal_type_{t}')(*args, **kwargs)

        stats = stats[['trade_dt','strategy_id','ret','bm','alpha',
        'pos_ret','trade_ret','pnl','pos_pnl','trade_pnl','mv',
        'fee','commission','buy','sell','trade_net','mv_ratio',
        'product_asset','turnover','update_time']]

        save_result(stats, '"public"."performance"', ['trade_dt', 'strategy_id'])

    return stats


if __name__ == "__main__":
    daily_statistics('20190708')
  

