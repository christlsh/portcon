import os
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10" 
os.environ["OMP_NUM_THREADS"] = "10" 
import time
import pandas as pd
import numpy as np
import polars as pl
import datetime as dt
import json
from gyqktoolkits.files import *
from gyqktools.utils import *
from gyqktools.files import *
from PortconOptimizers.optimizers import solve_opt_short_only
import quantage_mandate as mandate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-sdate',help='starting date',default=str(dt.datetime.now().date()))
parser.add_argument('-edate',help='ending date',default=str(dt.datetime.now().date()))
parser.add_argument('-mode',help='running mode',default='prod')
parser.add_argument('-profile_path',help='portfolio config profile json file',default='')
parser.add_argument('-risks_path',help='risk loadings path', default='/data/crypto/risk_model/CRY1/')
parser.add_argument('-srisk_path',help='srisk path', default='/data/crypto/risk_model/CRY1/srisk')
parser.add_argument('-cov_path',help='cov adj path', default='/data/crypto/risk_model/CRY1/cov')
parser.add_argument('-tc_path',help='tc model path', default='/data/live/portcon/tc/')
parser.add_argument('-trading_universe_path',help='trading universe path', default='/data/crypto/trading_universe_top30/')
parser.add_argument('-ptb_path', help='ptb path', default='/data/crypto/ptb/')
parser.add_argument('-rtb_path', help='ptb path', default='/data/return_table/30min/ret/')
parser.add_argument('-save_path',help='ar factor save path',default='/data/crypto/backtest/portcon/weights/')


def load_port_profile(profile_path):
    f = open(profile_path)
    return json.load(f)

def load_alpha(tdt,profile,db):
    alpha = pd.DataFrame()
    ct = 1
    while len(alpha) ==0:
        alpha = pd.DataFrame(list(db_prod[profile['alpha_name']].find({'datetime':dt.datetime.combine(tdt,dt.time(9,30,0))}))).drop(columns=['_id','ts'])
        print(str(dt.datetime.now())+' number of alpha is '+str(len(alpha)))
        if len(alpha) == 0:
            print(str(dt.datetime.now())+' alpha not available yet wait for 2 secs ...')
            time.sleep(2)
            ct+=1
        if ct ==100:
            print(str(dt.datetime.now())+' alpha not available and time out')
            break
    alpha = alpha.groupby('code').last()
    return alpha

def load_alpha_dev(tdt,itvl,profile):
    alpha = pd.read_parquet(os.path.join(profile['alpha_path'],str(tdt)+'.parquet'))
    if 'time' in alpha:
        pass
    else:
        alpha['time'] = alpha['datetime'].dt.time
    alpha = alpha[alpha['time']==itvl].copy()
    targets = []
    if "T1e" in profile.keys():
        targets = targets+[profile['T1e']]
    if "T2e" in profile.keys():
        targets = targets+[profile['T2e']]
    if "T4e" in profile.keys():
        targets = targets+[profile['T4e']]
    if "T8e" in profile.keys():
        targets = targets+[profile['T8e']]
    if "T24e" in profile.keys():
        targets = targets+[profile['T24e']]
        
    alpha = alpha[['code','datetime']+targets].copy()
    alpha = alpha.groupby('code').last()
    return alpha

#     preds = read_parquet(tdt-dt.timedelta(days=15),tdt, profile['alpha_path'])
# #     preds = preds[~preds.isin(ban_list)].copy()
#     pred_factors = [profile['target']]
#     preds = fill_data(preds, 'datetime','code',profile['target'], ffill = False, fill_value = 0)
#     preds= preds[preds['date']==tdt]
#     preds.drop(columns=['date'],inplace=True)
#     return preds.groupby('code').last()



def load_factor_exposure_dev(last_tdt,risks_path,profile):
    risk_factors = profile['style_factors'] + profile['industry_factors']
    risks = pd.DataFrame()
    for risk in risk_factors:
        temp = pd.read_parquet(os.path.join(risks_path,risk,str(last_tdt)+'.parquet'))
        if len(risks) == 0:
            risks = temp
        else:
            risks = pd.merge(risks, temp, on=['code','date'],how='outer')
    risks['date'] = tdt
    risks['time'] = dt.time(23,59,59)
    risks['datetime'] = risks.apply(lambda x: dt.datetime.combine(x.date,x.time), axis=1)
    risks.dropna(subset=profile['style_factors'], how='all',inplace=True)
    risks = risks.fillna(0.0)
    risks.sort_values(['datetime','code'],inplace=True)
    return risks

def load_srisk_dev(tdt,srisk_path):
    temp = pd.read_parquet(os.path.join(srisk_path,str(tdt)+'.parquet'))
    return temp[['code','srisk']]

def load_cov_dev(tdt, cov_adj_path):
    temp = pd.read_parquet(os.path.join(cov_adj_path,str(tdt)+'.parquet'))
    temp = temp.fillna(0.0)
    return temp

def load_tc(tdt,tc_path):
    return pd.read_parquet(os.path.join(tc_path,str(tdt)+'.parquet')).set_index('code')

def load_last_weights(last_tpt,weights_path):
    files = os.listdir(weights_path)
    files = [i for i in files if i.split('.')[-1]=='parquet']
    if len(files) == 0:
        print(str(dt.datetime.now())+' no weights infor available, use None')
        return None
    else:
        return pd.read_parquet(os.path.join(weights_path,str(last_tpt)+'.parquet'))

def query_open_price_dev(tdt,ptb_path):
    universe = list(get_all_securities(types='stock')['symbol'].unique())
    daily_ptb = pd.DataFrame()
    for i in range((len(universe)//200)+1):
        temp = get_price(universe[i*200:(i+1)*200], fields=['time','open','pre_close','volume'], start_date=tdt,end_date=tdt,fq=None, frequency='1d')
        daily_ptb = pd.concat([daily_ptb,temp],ignore_index=True)
    daily_ptb['open'] = daily_ptb['open'].fillna(daily_ptb['pre_close'])
    daily_ptb.rename(columns={'symbol':'code'},inplace=True)
    return daily_ptb[['code','open']].set_index('code')

def query_open_price_dev_backtest(tpt,ptb_path):
    tdt = tpt.date()
    itvl = tpt.time()
    ptb = pd.read_parquet(os.path.join(ptb_path,str(tdt)+'.parquet'))
    ptb['time'] = pd.to_datetime(ptb['datetime']).dt.time
    ptb = ptb[ptb['time']==itvl][['code','close']].copy()
    ptb.rename(columns={'close':'open'},inplace=True)
    return ptb[['code','open']].set_index('code')

def update_weights_at_open_dev(tpt,last_tpt,last_weights,AUM,ptb_path,ptb):
    if last_weights is None:
        return last_weights
    else:
        last_weights = last_weights
        rets = read_parquet(last_tdt,tdt, ptb_path,sort_cols=['datetime','code']).reset_index(drop=True)
        rets['pre_close'] = rets[['close','code']].groupby('code',group_keys=False)['close'].apply(lambda x: x.shift(1))
        rets['ret'] = (rets['close']-rets['pre_close'])/rets['pre_close']
        rets = rets[(rets['datetime']>last_tpt)&(rets['datetime']<=tpt)].copy()
        rets = rets.groupby('code').agg({'ret':'sum'}).reset_index()
        rets.rename(columns={'ret':'fret'},inplace=True)
        last_weights = pd.merge(last_weights,rets, on=['code'], how='left')
        last_weights.rename(columns={'weights':'weights_last',},inplace=True)
        last_weights.loc[:,'weights'] = last_weights['weights_last']*(1+last_weights['fret'])
        last_weights.drop(columns=['open'],inplace=True)
        last_weights = last_weights.merge(ptb.reset_index()[['code','open']], on=['code'], how='left')
        last_weights['weights'] = last_weights['weights'].fillna(last_weights['weights_last'])
    return last_weights.drop(columns=['fret'])

def filter_trading_universe(tdt,alpha, trading_universe_path):
    trading_universe = pd.read_parquet(os.path.join(trading_universe_path, str(tdt)+'.parquet'))
    alpha = alpha.loc[alpha.index.isin(trading_universe['code'])].copy()
    return alpha

def run_optimization(tpt,alpha,ptb,profile,last_weights,factor_exposure,srisks,cov_adj):
    time = tpt.time()
    total_cap = AUM
    target_vol = profile['portfolio_constraints']['target_vol']
    tc_const = profile['portfolio_constraints']['tc_const']
    turnover_limit = profile['portfolio_constraints']['turnover_limit']

    total_long_weights = profile['weights_constraints']['total_long_weights']
    total_short_weights = profile['weights_constraints']['total_short_weights']
    min_single_stock_weight = profile['portfolio_constraints']['min_single_stock_weight']
    swu_long = profile['portfolio_constraints']['single_stock_limit_long']
    swu_short = profile['portfolio_constraints']['single_stock_limit_short']

    factor_bounds = profile['factor_bounds']
    risks_factors = profile['style_factors']  + profile['industry_factors']
    control_factors = list(set(factor_bounds.keys()))
#     factor_bounds = {k:v for k,v in factor_bounds.items() if k in risks_factors}
    mandate.total_long_weights = total_long_weights
    mandate.total_short_weights = total_short_weights
    mandate.factor_bounds = factor_bounds
    cov_adj = cov_adj.loc[risks_factors,risks_factors].copy()
    if last_weights is None:
        codes = list(alpha.index)
    else:
        codes = list(set(list(alpha.index)+list(last_weights['code'])))
    # mandate.single_stock_upper_bounds = swu_long
    # mandate.single_stock_lower_bounds = swu_short
    lweights = pd.Series(index = codes,dtype='float64')
    uweights = pd.Series(index = codes,dtype='float64')

    lweights.loc[:] = -swu_short
    uweights.loc[:] = swu_long

    ret_1 = pd.Series(0.0, index=alpha.index)
    targets = []
    if "T1e" in profile.keys():
        ret_1 = ret_1 + alpha[profile['T1e']]*profile['portfolio_constraints']['T1e'] 
        targets = targets+[profile['T1e']]
    if "T2e" in profile.keys():
        ret_1 = ret_1 + alpha[profile['T2e']]*profile['portfolio_constraints']['T2e'] 
        targets = targets+[profile['T2e']]
    if "T4e" in profile.keys():
        ret_1 = ret_1 + alpha[profile['T4e']]*profile['portfolio_constraints']['T4e'] 
        targets = targets+[profile['T4e']]
    if "T8e" in profile.keys():
        ret_1 = ret_1 + alpha[profile['T8e']]*profile['portfolio_constraints']['T8e'] 
        targets = targets+[profile['T8e']]
    if "T24e" in profile.keys():
        ret_1 = ret_1 + alpha[profile['T24e']]*profile['portfolio_constraints']['T24e'] 
        targets = targets+[profile['T24e']]
    ret_1 = ret_1.dropna()
    with_preds = ret_1[ret_1!=0].index
    ret_1 = ret_1.loc[with_preds]
    preds_bottom_q = ret_1.quantile(0.5)

    
    if last_weights is not None:
        with_preds = [i for i in last_weights.code if i not in with_preds]
        with_preds = pd.Series(index=with_preds)
        with_preds.loc[:] = preds_bottom_q
        ret_1 = ret_1.append(with_preds)

    
    if last_weights is None:
        nancids = set(list(ret_1[ret_1.isna()].index))
        cids = list(set([i for i in ret_1.index if i in srisks['code'].to_list() ]))
        ret_1 = ret_1.loc[cids]

        factor_exposure = factor_exposure.loc[cids,risks_factors].fillna(0)
        last_weights_ = pd.DataFrame({'code':cids, 'weights':np.zeros(len(cids))})
        last_weights_ = last_weights_.set_index('code').loc[cids,'weights']

        ptb = ptb.loc[cids].copy()

        
        mandate.single_stock_lower_bounds = lweights.loc[cids]
        

        mandate.single_stock_upper_bounds = uweights.loc[cids]
        turnover_limit = 2
    else:
        nancids = set(list(ret_1[ret_1.isna()].index))
        cids = list(set([i for i in ret_1.index if i in srisks['code'].to_list() and i not in nancids ]))
        ret_1 = ret_1.loc[cids]
        # v21 = (v21 / total_cap * slimit).loc[cids]
        last_cids = last_weights.code.unique()
        to_add = [i for i in cids if i not in last_cids]
        to_add2 = [i for i in last_cids if i not in cids]
        
        if len(to_add2)>0:
            for cid in to_add2:
                ret_1.loc[cid] = preds_bottom_q
                srisks = pd.concat([srisks, pd.DataFrame({'code':[cid],'srisk':[srisks['srisk'].median()]})])
                # if cid not in tc.index:
                #     tc.loc[cid] = 3e-3
                # v21.loc[cid] = swu_long
            cids.extend(to_add2)
        if len(to_add) > 0:
            to_add = pd.DataFrame({'code':to_add})
            to_add['weights'] = 0
            last_weights = last_weights.append(to_add, ignore_index = 0)
        no_loadings = [i for i in cids if i not in factor_exposure.index]
        for cid in no_loadings:
            factor_exposure.loc[cid,risks_factors] = factor_exposure.loc[:,risks_factors].median()
        factor_exposure = factor_exposure.loc[cids,risks_factors].fillna(0)
        last_weights_ = last_weights.set_index('code').loc[cids,'weights']
        # left_tobuy = v21 * tlimit_const
        # v21_plus = last_weights_ + v21
        # v21_minus = last_weights_ - v21
        # left_tobuy[left_tobuy < 0] = 0 
        ptb = ptb.loc[cids].copy()


        mandate.single_stock_lower_bounds = lweights.loc[cids]

        mandate.single_stock_upper_bounds = uweights.loc[cids]
    # spec = pd.DataFrame(np.diag(np.square(srisks['srisk'].fillna(srisks['srisk'].median()))), columns = srisks['code'], index = srisks['code'])
    # power_number = profile['power_number']
    # dv_weight = pd.Series(np.power(np.sqrt(np.diag(spec)),power_number),index=spec.index)
    # ret_1 = ret_1/dv_weight.loc[ret_1.index]
    # ret_1 = ret_1/ret_1.std()
    # mandate.trading_costs = tc.loc[cids].values * tc_const
    mandate.trading_costs = pd.Series(0.0,index=cids).values
    factor_exposure['pred'] = ret_1

    try:
        
        weights = solve_opt_short_only(last_weights_, ret_1.values.astype('float64'), mandate.trading_costs, cov_adj, srisks.set_index('code').loc[cids], factor_exposure.astype('float64'), mandate, risks_factors, control_factors,turnover_limit,target_vol, min_single_stock_weight)
        if last_weights is None:
            weights['last_target_volume'] = 0
            weights['initial_weights'] = 0.0
        else:
            weights = weights.merge(last_weights[abs(last_weights['weights'])>0][['code','target_volume','weights']].rename(columns={'target_volume':'last_target_volume','weights':'last_weights'}), on=['code'], how='outer')
            weights['weights'] = weights['weights'].fillna(0.0)
            weights['last_target_volume'] = weights['last_target_volume'].fillna(0)
            weights['initial_weights'] = weights['last_weights']
            weights['initial_weights'] = weights['initial_weights'].fillna(0.0)
            weights.drop(columns=['last_weights'], inplace=True)
    except:
        try:
            weights = solve_opt_short_only(last_weights_, ret_1.values.astype('float64'), mandate.trading_costs, cov_adj, srisks.set_index('code').loc[cids], factor_exposure.astype('float64'), mandate, risks_factors, control_factors,turnover_limit*2,target_vol, min_single_stock_weight)
            if last_weights is None:
                weights['last_target_volume'] = 0
                weights['initial_weights'] = 0.0
            else:
                weights = weights.merge(last_weights[abs(last_weights['weights'])>0][['code','target_volume','weights']].rename(columns={'target_volume':'last_target_volume','weights':'last_weights'}), on=['code'], how='outer')
                weights['weights'] = weights['weights'].fillna(0.0)
                weights['last_target_volume'] = weights['last_target_volume'].fillna(0)
                weights['initial_weights'] = weights['last_weights']
                weights['initial_weights'] = weights['initial_weights'].fillna(0.0)
                weights.drop(columns=['last_weights'], inplace=True)
        except:
            try:
                weights = solve_opt_short_only(last_weights_, ret_1.values.astype('float64'), mandate.trading_costs, cov_adj, srisks.set_index('code').loc[cids], factor_exposure.astype('float64'), mandate, risks_factors, control_factors,turnover_limit*4,target_vol, min_single_stock_weight)
                if last_weights is None:
                    weights['last_target_volume'] = 0
                    weights['initial_weights'] = 0.0
                else:
                    weights = weights.merge(last_weights[abs(last_weights['weights'])>0][['code','target_volume','weights']].rename(columns={'target_volume':'last_target_volume','weights':'last_weights'}), on=['code'], how='outer')
                    weights['weights'] = weights['weights'].fillna(0.0)
                    weights['last_target_volume'] = weights['last_target_volume'].fillna(0)
                    weights['initial_weights'] = weights['last_weights']
                    weights['initial_weights'] = weights['initial_weights'].fillna(0.0)
                    weights.drop(columns=['last_weights'], inplace=True)
            except:
                weights = solve_opt_short_only(last_weights_, ret_1.values.astype('float64'), mandate.trading_costs, cov_adj, srisks.set_index('code').loc[cids], factor_exposure.astype('float64'), mandate, risks_factors, control_factors,turnover_limit*10,target_vol, min_single_stock_weight)
                if last_weights is None:
                    weights['last_target_volume'] = 0
                    weights['initial_weights'] = 0.0
                else:
                    weights = weights.merge(last_weights[abs(last_weights['weights'])>0][['code','target_volume','weights']].rename(columns={'target_volume':'last_target_volume','weights':'last_weights'}), on=['code'], how='outer')
                    weights['weights'] = weights['weights'].fillna(0.0)
                    weights['last_target_volume'] = weights['last_target_volume'].fillna(0)
                    weights['initial_weights'] = weights['last_weights']
                    weights['initial_weights'] = weights['initial_weights'].fillna(0.0)
                    weights.drop(columns=['last_weights'], inplace=True)

           
    weights['target_dvol'] = (weights['weights']) * total_cap
    weights = pd.merge(weights, ptb.reset_index()[['code','open']], on = ['code'], how = 'inner')
    weights.loc[~weights['open'].isna(),'target_volume']= (weights[~weights['open'].isna()]['target_dvol'] / weights[~weights['open'].isna()]['open'] ).apply(round) 
    weights['trade'] = weights['target_volume'] - weights['last_target_volume']
    weights.loc[~weights['open'].isna(),'weights'] = weights.loc[~weights['open'].isna(),'target_volume'] * weights.loc[~weights['open'].isna(),'open'] / total_cap
    weights.loc[~weights['open'].isna(),'trade_weights'] = weights.loc[~weights['open'].isna(),'trade'] * weights.loc[~weights['open'].isna(),'open'] / total_cap

 
    weights = weights[(abs(weights['weights'])>0)|(abs(weights['initial_weights'])>0)].copy()
    weights['datetime'] = tpt
    weights.to_parquet(os.path.join(weights_path,str(tpt)+'.parquet'))
    return True

if __name__ == "__main__":
    args = parser.parse_args()
    sdate = args.sdate
    edate = args.edate
    profile_path = args.profile_path
    tc_path = args.tc_path
    ptb_path = args.ptb_path
    save_path = args.save_path
    risks_path = args.risks_path
    srisk_path = args.srisk_path
    cov_path = args.cov_path
    trading_universe_path = args.trading_universe_path
    intervals = pl.time_range(dt.time(0,0,0),dt.time(23,59,59),interval='1h', closed='left', eager=True).to_list()
    dts = pl.datetime_range(utils.to_date(sdate), dt.datetime.combine(utils.to_date(edate), dt.time(23, 59, 59)), interval='1d', closed="left", eager=True).dt.date().to_list()
    trading_points = []
    for tdt in dts:
        for itvl in intervals:
            trading_points.append(dt.datetime.combine(tdt,itvl))
    trading_points.sort()
    profile = load_port_profile(profile_path)
    AUM = profile['portfolio_constraints']['total_cap']
    weights_path = os.path.join(save_path,profile['strat_name'])
    mkdir(os.path.join(weights_path))
    files = os.listdir(weights_path)
    if len(files)!=0:
        tpts = [dt.datetime.fromisoformat(i.split('.')[0]) for i in files]
        last_tpt = max(tpts)
        trading_points = [i for i in trading_points if i >last_tpt]
    else:
        last_tpt = None
    for tpt in trading_points:
        tdt = tpt.date()
        itvl = tpt.time()
        print(str(dt.datetime.now())+' '+str(tdt)+' '+str(itvl))
        last_tdt = tdt-dt.timedelta(days=1) 
        factor_exposure = load_factor_exposure_dev(last_tdt,risks_path,profile)
        factor_exposure.set_index('code',inplace=True)
        srisks = load_srisk_dev(tdt,srisk_path)
        srisks.loc[srisks['srisk']<1e-3,'srisk'] = srisks['srisk'].median()
        
        cov_adj = load_cov_dev(tdt, cov_path)
        # tc_model = load_tc(tdt,tc_path)
        
        alpha = load_alpha_dev(tdt,itvl,profile)
        
        alpha = filter_trading_universe(tdt, alpha, trading_universe_path)
        ptb = query_open_price_dev_backtest(tpt,ptb_path)
        if last_tpt == None:
            last_weights = None
        else:
            last_weights = load_last_weights(last_tpt,weights_path)
            
            last_weights = update_weights_at_open_dev(tpt,last_tpt,last_weights,AUM,ptb_path,ptb)
        
        
        print(str(dt.datetime.now()))
        run_optimization(tpt,alpha,ptb,profile,last_weights,factor_exposure,srisks,cov_adj)
        print(str(dt.datetime.now()))
        last_tpt = tpt