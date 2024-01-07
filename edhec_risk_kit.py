import pandas as pd
import scipy.stats
import numpy as np

## Fonction get pour charger les données
def get_ind_returns():

    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    
    #On convertit les dates en un format de date
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    
    #Les noms des columns comportent des espaces, il faut les supprimer
    ind.columns = ind.columns.str.strip()
    
    return ind

def get_ffme_returns():

    """Load the Fama-French Dataset"""
    
    me_n = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    
    rets = me_n[["Lo 10", "Hi 10"]]
    rets=rets/100
    rets.columns=["SmallCap","LargeCap"]
    rets.index=pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets

def get_hfi_returns():

    """Load the EDHEC hedge Fund Index Return Dataset"""
    
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                       header=0, index_col=0, na_values=-99.99, parse_dates=True)
    
    rets=hfi/100
    rets.index=rets.index.to_period('M')
    return rets

def get_ind_returns():
    
    """Load and format the Ken French 30 Industry Portfolio Value Weighted Monthly Retunrs
    """
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100
    #On convertit les dates en un format de date
    ind.index=pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    #Les noms des columns comportent des espaces, il faut les supprimer
    ind.columns = ind.columns.str.strip()
    return ind

##Fonctions pour calculer des métriques

def wealth_index(return_series: pd.Series):
    
    wealth_index_=1000*(1+return_series).cumprod()
                 
    return wealth_index_


def previous_peaks(return_series: pd.Series):  
    
    wealth_index_=wealth_index(return_series)
    previous_peaks_=wealth_index_.cummax()
    
    return previous_peaks_


def drawdowns(return_series: pd.Series):
    
    
    wealth_index_=wealth_index(return_series)
    previous_peaks_=previous_peaks(return_series)
    
    drawdowns_=(wealth_index_-previous_peaks_)/previous_peaks_

    return drawdowns_


def semideviation(r):
    
    is_negative = r < 0
    return r[is_negative].std(ddof=0)
    

def skewness(r):
    
    X=r-r.mean()
    s=r.std(ddof=0)
    E=(X**3).mean()
    
    return E/s**3

def kurtosis(r):
    
    X=r-r.mean()
    s=r.std(ddof=0)
    E=(X**4).mean()
    
    return E/s**4

def is_normal(r, level=0.01):
    """ to test the normality of a series"""
    
    """ Appliying Jarque Bera test """
    
    """ with p-value 1%"""
    
    statistic, p_value= scipy.stats.jarque_bera(r)
    
    return p_value > level

##Calcul des value at risk

from scipy.stats import norm

def var_gaussian(r, level=5, modified=False):
    z=norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z=(z +
           (z**2-1)*s/6 +
           (z**3-3*z)*(k-3)/24 -
           (2*z**3-5*z)*(s**2)/36
          )
        
    return -(r.mean()+z*r.std(ddof=0))

def cvar_historic(r , level=5):
    
    if isinstance(r, pd.Series):
        is_beyond = r <= r.quantile(q=level/100)
        return r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or a DataFrame")

        
## Calcul de metrique annualisée (moyenne, ecart type, sharpe-ratio) c
        
def annualize_expected_returns(r, periods_per_year):
    
    """
    Annualizes a set of returns, periods are infer per year
    On calcule le rendement moyen (moyenne géométrique) et on le transforme en rendement moyen annuel
    """        
    compounded_growth=(1+r).prod()
    n_periods=r.shape[0]-r.isnull().sum() #On évite les valeurs NaN
    
    return compounded_growth**(periods_per_year/n_periods)-1
    
def annualize_vol(r, periods_per_year):
    
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    
    """ computes the annualized sharpe ratio of a set of returns
    """
    
    #comvert the annual riskfree rate to a riskfree rate per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    
    #Compute excess return 
    excess_ret = r - rf_per_period
    
    #convert data per period to annual
    ann_ex_ret = annualize_expected_returns(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    
    return ann_ex_ret/ann_vol

def portfolio_return(weight, returns):
    """Permet de calculer l'espérance d'un portefeuille via la somme pondérée des espérances de chaque actifs (le vecteur returns contient des espérances. Attention à la période 
    """
    return weight.T @ returns

def portfolio_volatility(weight, covmat):
    """Pour calculer la variance d'un portefeuille. Prend en entrée la matrice de covariance
    """
    return (weight.T @ covmat @ weight)**(0.5)

def minimize_vol(target_return, er, cov):
    """
    target_returns -> weight vector
    """
    from scipy.optimize import minimize
    #Nombre de données
    n = er.shape[0]
    
    #Données initiales pour le Weight vector (W)
    init_guess =np.repeat(1/n,n)
    
    #Données limites pour W, il faut n tuples
    bounds = ((0.0, 1.0),)*n
    
    #Contrainte sur le rendement souhaité
    return_is_target = {
        'type': 'eq',
        'args':(er,),
        'fun': lambda w,er:target_return - portfolio_return(w,er) 
    }
    
    #Contrainte sur la somme des poids
    weights_sum_to_1={
        'type': 'eq',
        'fun': lambda w:np.sum(w)-1
    }
        
    results=minimize(portfolio_volatility, init_guess, 
                args=(cov,), method="SLSQP",
                options={'disp':False},
                constraints=(return_is_target, weights_sum_to_1),
                bounds=bounds
              )
    return results.x
  
def optimal_weights(er, cov,n_points):
    
    target_rs=np.linspace(er.min(), er.max(), n_points)
    w=[minimize_vol(t,er,cov) for t in target_rs]
    return w

def max_sharpe_ratio(riskfree_rate, er, cov):
    """
    RiskFree rate + ER + COV -> weight vector
    """
    from scipy.optimize import minimize
    #Nombre de données
    n = er.shape[0]
    
    #Données initiales pour le Weight vector (W)
    init_guess =np.repeat(1/n,n)
    
    #Données limites pour W, il faut n tuples
    bounds = ((0.0, 1.0),)*n
    
    #Contrainte sur la somme des poids
    weights_sum_to_1={
        'type': 'eq',
        'fun': lambda w:np.sum(w)-1
    }
    
    def negative_sharpe_ratio(weights, riskfree_rate, er, cov):
        r=portfolio_return(weights, er)
        vol=portfolio_volatility(weights, cov)
        return - (r-riskfree_rate)/vol
    
    results=minimize(negative_sharpe_ratio, init_guess, 
                args=(riskfree_rate,er,cov,), method="SLSQP",
                options={'disp':False},
                constraints=(weights_sum_to_1),
                bounds=bounds
              )
    return results.x

def gmv(cov):
    
    """ Return the weight of the global minimum variance potfolio"""
    
    n=cov.shape[0]
    return max_sharpe_ratio(0, np.repeat(1,n), cov)
    
def plotting_efficient_frontier(er, cov, n_points, show_cml=False, style='.-', riskfree_rate=0, show_ew=False, show_gmv=False):
  
    """Fonction pour dessiner l'efficient frontier de N actifs.
    """
    
    weights=optimal_weights(er, cov,n_points)
    rets=[portfolio_return(w,er) for w in weights]
    vol=[portfolio_volatility(w,cov) for w in weights]
    ef=pd.DataFrame({"R":rets, "Vol":vol})
    
    
    ax=ef.plot.line(x="Vol", y="R", style=style)
    ax.set_xlim(left=0)
    
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        vol_ew=portfolio_volatility(w_ew, cov)
        ax.plot([vol_ew],[r_ew], color="goldenrod", marker="o", markersize=10)
                
    if show_cml:
        w_msr=max_sharpe_ratio(riskfree_rate, er, cov)
        r_msr=portfolio_return(w_msr, er)
        vol_msr=portfolio_volatility(w_msr, cov)
        #add capital market line
        cml_x=[0, vol_msr]
        cml_y=[riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed")
    
    if show_gmv: #global minimum variance
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv, er)
        vol_gmv=portfolio_volatility(w_gmv, cov)
        ax.plot([vol_gmv],[r_gmv], color="midnightblue", marker="o", markersize=10)
    return ax
