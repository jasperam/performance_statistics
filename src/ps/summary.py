#
# summary.py
# @author Neo Lin
# @description check and statistics performance
# @created 2019-09-23T11:28:55.643Z+08:00
# @last-modified 2020-03-16T20:54:25.381Z+08:00
#
import pandas as pd
import numpy as np

from jt.utils.db import PgSQLLoader, SqlServerLoader
from jt.utils.calendar import TradeCalendarDB
from ps.data_loader import get_strategy_info, get_manager_info, get_benchmark_info, get_deduction, \
    get_performance, get_history_alpha, get_cumulative_bonus, get_defer_bonus, get_daily_alpha, get_argo_pnl
from ps.utils import save_result

NAVDB = SqlServerLoader('trade71')
ATTDB = PgSQLLoader('attribution')
CALENDAR = TradeCalendarDB('jtder')
TODAY = CALENDAR.get_trading_date()

def check_pnl_with_nav(from_=None, to_=TODAY):
    sql = f'''
        SELECT trade_dt,Account,AccountName,dailyPnL,TotalAsset,NetAsset,preNetAsset,prefee_PnL,dailyfee_PnL,afterfee_PnL
        FROM [dbo].[JasperNAVData] where trade_dt <= '{to_}'
    '''
    if not from_ is None:
        sql = sql + f" and trade_dt>='{from_}'"
    nav = NAVDB.read(sql)


def cal_jd_performance(from_='20190701', to_=TODAY):
    sql = f'''
        select * from "allocation" 
    '''
    alloc = ATTDB.read(sql)
    jds_id = "'"+"','".join(alloc.loc[alloc['strategy_name'].str.contains('JD'), 'strategy_id'].to_numpy())+"'"

    sql = f'''
        SELECT product_id,sum(pnl) as pnl,sum(stock_mv) as stock_mv,sum(bm_pnl) as bm_pnl
        from (
        SELECT left(a.product_id,2) as product_id,b.strategy_id,trade_dt,pnl,stock_mv,bm_pnl
        FROM "performance" b LEFT JOIN "allocation" a on a.strategy_id = b.strategy_id 
        where trade_dt between '{from_}' and '{to_}' and b.strategy_id in ({jds_id}) ) as tst
        GROUP BY product_id order by product_id
    '''
    jd_per = ATTDB.read(sql)
    adjust_pct = pd.DataFrame({'product_id':['55','64','79','80','84','85','91','93'],
                                'adjust' : [0, 0.07, 0, 0.07, 0, 0.05, 0.07, 0]})
    jd_per = jd_per.merge(adjust_pct, how='left').fillna(0)
    jd_per['bm_adjust'] = jd_per['stock_mv'] * jd_per['adjust'] / 245
    natural_acc = ['55','79','93']
    jd_per.loc[jd_per['product_id'].isin(natural_acc),'bm_pnl'] = 0
    jd_per['adjust_pnl'] = jd_per['pnl'] - jd_per['bm_pnl'] - jd_per['bm_adjust']
    return jd_per

def get_benchmark_rct():
        bm = get_benchmark_info()
        def _inner(f_):
            for x, y in zip(bm['bm_id'], bm['bm_adjust_return'].astype(str)):               
                f_ = f_.replace(x, y)
            return eval(f_)/245
        return _inner

def cal_monthly_performance(from_='20190701', to_=TODAY):  

    def get_compliance_boundary():
        level_boundary = {
            'high': [[5.7, 3.3, 2.4, 1.7], [10.7, 6.2, 4.4, 3.1]],
            'mid': [[4.4, 2.5, 1.8, 1.3], [8.2, 4.8, 3.4, 2.4]],
            'low': [[3.4, 2.0, 1.4, 1.0], [6.3, 3.7, 2.6, 1.8]],
        }
        def _inner(df):
            b = level_boundary.get(df['level'], NotImplementedError)
            uplimit = b[1][0] if df['days']<=20 else (b[1][1] if df['days']<=60 else (b[1][2] if df['days']<=120 else b[1][3]))
            downlimit = b[0][0] if df['days']<=20 else (b[0][1] if df['days']<=60 else (b[0][2] if df['days']<=120 else b[0][3]))      
            return downlimit, uplimit
        return _inner

    def cal_compliance_rate():
        def _inner(df):
            _r = (df['sr']-df['boundary'][0])/(df['boundary'][1]-df['boundary'][0])
            _r = 0 if _r<0 else (_r if _r<1 else 1)
            return _r
        return _inner
            
    def cal_dividend_rate():
        div_boundary = {
            '1': (0.03, 0.04),
            '2': (0.04, 0.06),
            '3': (0.06, 0.09)
        }
        def _inner(df):
            b = div_boundary.get(df['dividend_type'], (0, 0))
            return b[0] + df['compliance_rate']*(b[1]-b[0])
        return _inner
    

    # statistics pnl, bm_pnl and bm_adjust_pnl   
    per = get_performance(from_, to_)
    strategy = get_strategy_info()
    manager = get_manager_info()
    per = per.merge(strategy.loc[:,['strategy_name','manager_id','bm']], how='left', on='strategy_name')
    per = per.merge(manager.loc[:,['manager_id','manager_name','dividend_type']], how='left', on='manager_id')
    per['bm_adjust_rate'] = per['bm'].apply(get_benchmark_rct())
    per['bm_adjust'] = per['stock_mv'] * per['bm_adjust_rate']
    per.set_index('strategy_name', inplace=True)
    
    # calculation turnover, sr, day cnts
    # The same strategy is weighted according to the market value of the position to get the adjustment alpha of the T day, in the calculation of sr
    
    history = get_history_alpha(from_, to_)
    # calculation turnover
    turnover = history.loc[:,['strategy_name','turnover']].groupby('strategy_name').mean() * 245
    turnover['level'] = turnover['turnover'].apply(lambda x: 'high' if x > 100 else ('mid' if x > 25 else 'low'))
    # deal with special strategy 
    turnover.loc['ARBT','level'] = 'mid'
    turnover.loc['FANCTA100','level'] = 'mid'
    turnover.loc['ZS','level'] = 'mid'
    turnover.loc['MJCTA','level'] = 'high'
    turnover.loc['MJOPT','level'] = 'high'
    
    # calculation sr
    sr = history.loc[:,['strategy_name','alpha']].groupby('strategy_name').std() * 245**0.5 /10000    
    sr.rename(columns={'alpha':'vol'}, inplace=True)
    sr = sr.merge(turnover, left_index=True, right_index=True)
    ann_alpha = history.loc[:,['strategy_name','alpha']].groupby('strategy_name').mean() * 245 /10000
    sr = sr.merge(ann_alpha, left_index=True, right_index=True)
    sr = sr.merge(per, left_index=True, right_index=True, how='inner') 
    sr['sr'] = (sr['alpha']-sr['bm_adjust_rate']*245)/sr['vol'] 

    days = history.loc[:,['strategy_name','trade_dt']].groupby('strategy_name').count()
    days.rename(columns={'trade_dt':'days'}, inplace=True)
    sr = sr.merge(days, left_index=True, right_index=True)
    deduction = get_deduction(from_, to_)
    deduction.set_index('strategy_name', inplace=True)
    sr = sr.merge(deduction, left_index=True, right_index=True, how='left').fillna(0)
    
    # get argo pnl
    argo_pnl = get_argo_pnl(from_, to_).set_index('strategy_name')
    sr = sr.merge(argo_pnl, left_index=True, right_index=True, how='left').fillna(0)
    sr['adjust_pnl'] = sr['pnl'] - sr['bm_pnl'] - sr['bm_adjust'] - sr['deduction'] - sr['argo_pnl']
    
    # cal compliance rate
    sr['boundary'] = sr.apply(get_compliance_boundary(), axis=1)    
    sr['compliance_rate'] = sr.apply(cal_compliance_rate(), axis=1)    
    sr['dividend_rate'] = sr.apply(cal_dividend_rate(), axis=1)   
    sr['bonus'] = sr.apply(lambda x: 
        x['adjust_pnl']*x['dividend_rate']*x['compliance_rate'] if x['adjust_pnl']>0 else x['adjust_pnl']*x['dividend_rate'], axis=1)
    sr['start_dt'] = from_
    sr['end_dt'] = to_
        
    # save_result(sr.reset_index(), ATTDB, '"public"."statistics_detail"', ['start_dt', 'end_dt', 'strategy_name'])
    
    # cal every pm's performance
    bonus = sr.loc[:,['bonus','manager_id','manager_name']]
    dv_record = bonus.loc['DV']
    dv_record['bonus'] = dv_record['bonus']/2
    dv_record['manager_id'] = '06'
    dv_record['manager_name'] = 'Peter'
    dv_record.name = 'DV1'
    
    # special deal to DV 0.5->Dox, 0.5->Peter   
    bonus.loc['DV','bonus']=bonus.loc['DV','bonus']/2
    bonus = bonus.append(dv_record)
    bonus = bonus.groupby('manager_name')[['bonus']].sum()
    bonus.rename(columns={'bonus':'cumulative_bonus'}, inplace=True)

    # save to DB
    cumulative_bonus = get_cumulative_bonus()
    his_bonus = cumulative_bonus.loc[cumulative_bonus['trade_dt']<to_,['manager_name','bonus']].groupby('manager_name').sum()
    his_bonus.rename(columns={'bonus':'history_bonus'}, inplace=True)
    bonus = bonus.merge(his_bonus, left_index=True, right_index=True, how='left').fillna(0)
    bonus['trade_dt'] = to_
    bonus['bonus'] = bonus.apply(lambda x:x['cumulative_bonus']-x['history_bonus'] if x['cumulative_bonus']-x['history_bonus']>0 else 0, axis=1)
    bonus['current_redemption'] = bonus['bonus'] * 0.8
    bonus['defer'] = bonus['bonus'] * 0.2
        
    # get defer bonus
    defer_bonus = get_defer_bonus(to_)
    
    return bonus, defer_bonus, sr

def save_monthly_stat(from_, to_):
    bonus, defer_bonus, sr = cal_monthly_performance(from_=from_, to_=to_)
    save_result(bonus.loc[:,['trade_dt','cumulative_bonus','bonus','current_redemption','defer']].reset_index(), ATTDB, 
        '"public"."bonus"', ['trade_dt', 'manager_name'])
    

def daily_report(date_):
    per = get_performance(date_, date_)
    strategy = get_strategy_info()   
    per = per.merge(strategy.loc[:,['strategy_name','bm']], how='left', on='strategy_name')   
    per['bm_adjust_rate'] = per['bm'].apply(get_benchmark_rct())
    per['bm_adjust_pnl'] = per['stock_mv'] * per['bm_adjust_rate']
    
    history = get_daily_alpha(date_)
    per = per.merge(history, on='strategy_name', how='left').fillna(0)
    per['adjust_pnl'] = per['pnl'] - per['bm_pnl'] - per['bm_adjust_pnl']

    all_per = get_performance('20190701', date_)
    all_per.set_index('strategy_name', inplace=True)
    all_history = get_history_alpha('20190701', date_)
    # calculation sr
    sr = all_history.loc[:,['strategy_name','alpha']].groupby('strategy_name').std() * 245**0.5 /10000    
    sr.rename(columns={'alpha':'vol'}, inplace=True)    
    ann_alpha = all_history.loc[:,['strategy_name','alpha']].groupby('strategy_name').mean() * 245 /10000
    sr = sr.merge(ann_alpha, left_index=True, right_index=True)
    sr = sr.merge(all_per, left_index=True, right_index=True, how='inner') 
    days = all_history.loc[:,['strategy_name','trade_dt']].groupby('strategy_name').count()
    days.rename(columns={'trade_dt':'days'}, inplace=True)
    sr = sr.merge(days, left_index=True, right_index=True)    
    sr = sr.merge(strategy.loc[:,['strategy_name','bm']], how='left', on='strategy_name')  
    sr['bm_adjust_rate'] = sr['bm'].apply(get_benchmark_rct())
    sr['bm_adjust_rate'] = sr['bm_adjust_rate'] * 245
    sr['sr'] = (sr['alpha']-sr['bm_adjust_rate'])/sr['vol'] 
    per = per.loc[:,['strategy_name','alpha','pnl','bm_pnl','bm_adjust_pnl','adjust_pnl','turnover']]
    per.sort_values('strategy_name' ,inplace=True)
    sr = sr.loc[:,['strategy_name','days','sr','alpha','vol','bm_adjust_rate']]
    sr.sort_values('strategy_name',inplace=True)
    return per, sr


if __name__ == "__main__":
    # jd_per = cal_jd_performance(from_='20200101', to_='20200131')
    # jd_per.to_csv(r'E:\temp\jd01.csv')    
    # bonus, defer_bonus, sr = cal_monthly_performance(from_='20190701', to_='20191231')
    # bonus.to_csv(r'e:\temp\bonus.csv')
    # defer_bonus.to_csv(r'e:\temp\defer_bonus.csv')
    # sr.to_csv(r'e:\temp\sr.csv')
    # print(sr)
    per, sr = daily_report('20200313')
    # print(per)
    # per.to_csv(r'e:\temp\per.csv')
    # sr.to_csv(r'e:\temp\sr.csv')
    # save_monthly_stat('20190701','20200131')