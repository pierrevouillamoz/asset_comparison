import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import edhec_risk_kit as erk

data_gold=pd.read_csv("data/amundi_gold.csv", sep=',', parse_dates=["Date"], index_col=0)
data_msciw=pd.read_csv("data/msci_world.csv", sep=',', parse_dates=["Date"], index_col=0)
data_bonds=pd.read_csv("data/G7_bonds_FTSE.csv", sep=',', parse_dates=["Date"], index_col=0)

data=pd.concat([data_gold,data_msciw,data_bonds], axis=1, join='inner')
data.columns=["gold","msciw","bonds"]


def backtest_dca(data, initial_amount=5000, dca=250, r_wished=0.1, r_free=0.01, late=5, asset='msciw'):
    
    """
    This function generates several portfolios through a timeline like a real investment routine.
    The goal is to back test the parameters of a dollar cost average (DCA) strategy.
    The algorithm is a time loop (monthly based). At each step, different portfolio values are updated and at some date (yearly) 
    the optimal allocation is computed.
    
    Several tactics are tested : 
    - the portfolio allocation is computed each year - i.e. rolling optimization
    - the portfolio allocation is computed once a time - i.e. fixed allocation or optimization
    - investment is made on a singke risk free asset with the usual risk free asset rate (set by user)
    - investment is made on a single risk free asset with the expected rate (set by user) - utopical situation
    - investment is made on a single asset (set by user)
    
    Inputs :
    - data : the value of assets
    - r_wished : the desired expected return
    - r_free : risk free rate
    - initial_amount : amount of the first investment
    - dca : amount invested each month
    - late : the duration (year) before the first investment and the number of years taken into account to compute the yearly 
    reallocation
    - asset : asset selected for the single asset tactic
        
    Outputs :
    - W : the different weight vector for the rolling optimization strategy - type DataFrame
    - Wfixed_allocation : the unique weight vector for the fixed allocation strategy - type Series
    - PF : portfolio for the rolling optimization strategy - type DataFrame
    - PF_FixedAllocation : portfoilo for the fixed allocation strategy - type DataFrame
    - DCAmin : portfolio with only risk free asset - type Series
    - DCAmax : portfolio with only utopical risk free asset at desired rate - type Series
    - DCAasset : portfolio with only one asset defined by the user - type Series
    - total_invested : total amount invested, 1st and each month - type float
     
    
    """
    
    nb_asset=len(data.columns)
    
    data_pct=data.pct_change() 
    #Please note, we compute the percentage variation between two values at timestep i and timestep i-1. So data_pct.iloc[0]=nan   
    #i corresponds to the beginning of month i.
          
    #Features updated at each time step
    w=np.zeros((nb_asset)) #vecteur poids du portefeuille de la stratégie
    pf=np.zeros((nb_asset)) #portefeuille en temps réel de la stratégie testée
    dca_min=0  #type nombre
    dca_max=0  #type nombre
    dca_asset=0 #achat d'un même asset en dca
    pf_FixedAllocation =np.zeros((nb_asset))  #Portefeuille en temps réel quand la stratégie est l'allocation fixe
    total_invested=0 #montant total investit 
    
    #Container features to store updated features at each time step
    W=[] #stockage des allocations
    PF=[] #stockage du portefeuille
    DCAmin=[]  #stockage du portefeuille dca sur act
    DCAmax=[]  #DCA si actif sans risque au taux souhaité
    DCAasset=[]  #On tente d'accheter toujours le même asset stockage de cette valeur
    PF_FixedAllocation=[] #stockage du portefeuille à allocation fixe
    Annee=[]  #Stockage de l'année où à lieu la nouvelle allocation
    

    #Optimal allocation for a fixed allocation at one time. The optimization happened after a period of "late" years
    covmat=data_pct.iloc[1:late*12+1].cov()*12
    er=erk.annualize_expected_returns(data_pct.iloc[1:late*12+1],12)
    Wfixed_allocation=erk.minimize_vol(r_wished,er, covmat) #The weight of the fixed allocation
    
    
    #time loop
    for i in range(0,len(data_pct)):
        
        #while for investment is not happened, no uptdate, nor compute nor investment are done
        #we respect a timedelta (feature late) before doing first allocation and
        
        # 1 Our different portfolio values are updated each month
        if i>late*12:
            pf=pf * (1+data_pct.iloc[i])  #The update of the portfolio requires only asset variation
            pf_FixedAllocation=pf_FixedAllocation *(1+data_pct.iloc[i])
            dca_asset= dca_asset * (1+data_pct[asset].iloc[i])
            dca_min=dca_min*(1+r_free)**(1/12)
            dca_max=dca_max*(1+r_wished)**(1/12)

        # 2 The first allocation and investment
        if i==late*12:
               
            #First investments   
            dca_asset=initial_amount
            DCAasset.append(dca_asset)

            dca_min=initial_amount
            DCAmin.append(dca_min)

            dca_max=initial_amount
            DCAmax.append(dca_max)            
                
            total_invested+=initial_amount
            
            #First computing of the rolling optimisation
            X=data_pct.iloc[i-late*12+1:i+1]  
            #the current month should be taken into account in the computing, because return(i) = price(i) - price(i-1)
            #We use i+1 as last term of interval is not taken into account in Python
            er=erk.annualize_expected_returns(X,12) #we compute expected return
            covmat=X.cov()*np.sqrt(12) #we compute covariance matrix
            w=erk.minimize_vol(r_wished,er,covmat) #optimal weight are computed throught the minimize_vol algorithm

            W.append(w) #Optimal weight at this step is stored in W
            Annee.append(data_pct.iloc[i].name.strftime("%m-%Y")) #This list store the current year
            
            #Reallocation are made
            pf=w*initial_amount   #For the portfolio of the rolling strategy   
            pf_FixedAllocation = Wfixed_allocation * initial_amount #For the fixed allocation portfolio
            
            #We store the two portfolios current values in two list 
            PF.append(list(pf))
            PF_FixedAllocation.append(list(pf_FixedAllocation))
    
        # 2bis New optimization computing and yearly reallocation
        if (i%12==0 and i>late*12):
            
            #On s'interesse à la valeur du pf que l'on va rerépartir
            pf_value=pf.sum()  
            pf_FixedAllocation_value=pf_FixedAllocation.sum()
            
            #new optimization for the rolling allocation portfolio
            X=data_pct.iloc[i-late*12+1:i+1]  
            er=erk.annualize_expected_returns(X,12)
            covmat=X.cov()*12
            w=erk.minimize_vol(r_wished,er,covmat) #new optimal weights

            W.append(w) #store of new weights
            Annee.append(data_pct.iloc[i].name.strftime("%m-%Y"))
            
            #Reallocation
            pf=w*pf_value   #For the rolling optimization portfolio   
            pf_FixedAllocation = Wfixed_allocation * pf_FixedAllocation_value #For the fixed allocation portfolio
        
        # 3 Current investment of the month - also work if dca=0
        if i>late*12:
                    
            pf=pf+dca*w  #We invest each month dca splitted according w (we assume asset can be fractionnable)
            PF.append(list(pf))

            pf_FixedAllocation = pf_FixedAllocation + dca * Wfixed_allocation
            PF_FixedAllocation.append(list(pf_FixedAllocation))

            dca_asset+=dca
            DCAasset.append(dca_asset)

            dca_min+=dca
            DCAmin.append(dca_min)

            dca_max+=dca
            DCAmax.append(dca_max)

            total_invested+=dca

    
    #Outputs formatting
    
    W=pd.DataFrame(W, index=Annee, columns=data_pct.columns)
    Wfixed_allocation=pd.Series(Wfixed_allocation, index=data_pct.columns)
    PF=pd.DataFrame(PF,index=data_pct.iloc[late*12:].index, columns=data_pct.columns)
    PF_FixedAllocation=pd.DataFrame(PF_FixedAllocation, index=data_pct.iloc[late*12:].index, columns=data_pct.columns)
    
    DCAmin=pd.Series(DCAmin, index=data_pct.iloc[late*12:].index, name="DCAmin")
    DCAmax=pd.Series(DCAmax, index=data_pct.iloc[late*12:].index, name="DCAmax")
    DCAasset=pd.Series(DCAasset, index=data_pct.iloc[late*12:].index, name="DCAasset")
        
    return W, Wfixed_allocation, PF, PF_FixedAllocation, DCAmin, DCAmax, DCAasset, total_invested
