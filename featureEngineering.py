import pandas as pd
import numpy as np
class featureEngineering_TrainTest:
    '''
    Parameters
    1) Training Data Set
   
    Returns :
    Feature Engineered Variables for the training dataset
    '''
    def __init__(self, trainingData):
        self.trainingData = trainingData
        
    
    def featureEngineering(self):
        df = self.trainingData
        df['growth_past'] = ((df['Volume_2019'] - df['Volume_2018']))
        df['industry_growth'] = [np.mean(df.loc[(df['segment']==df['segment'][i])  & (df['poc_image']==df['poc_image'][i]) & (df['sdfc_Tier'] == df['sdfc_Tier'][i]) & (df['sub_segment']==df['sub_segment'][i]) & (df['province']==df['province'][i]) , "growth_past" ]) for i in range(len(df['growth_past']))]
        df['market_cap'] = [df['Volume_2019'][i]/(np.sum(df.loc[  (df['segment'] == df['segment'][i]) & (df['sub_segment']==df['sub_segment'][i]) & (df['poc_image']==df['poc_image'][i]) & (df['province']==df['province'][i]) & (df['sdfc_Tier']==df['sdfc_Tier'][i]) , "Volume_2019"] )) for i in range(len(df['industry_growth']))]
        df['market_cap'] = df['market_cap'].fillna(0)
    
    
    
        df['future_growth'] = [(df['growth_past'][i] + df['market_cap'][i]*df['industry_growth'][i])/(1+df['market_cap'][i]) for i in range(len(df['industry_growth']))]
    
    
        df['order_size'] = [df['Volume_2019 Product'][i]/(np.amax(df.loc[df['Product Set']==df['Product Set'][i] , "Volume_2019 Product"])) for i in range(len(df['growth_past']))]
    
    
        val = df['GTO_2019'] * ((df['Volume_2019'] + df['future_growth'])/(df['Volume_2019']+0.01))
        df['Expected_GTO'] = val
        df['Expected_product_volume'] = df['Volume_2019 Product']*((df['Volume_2019'] + df['future_growth'])/(df['Volume_2019']+0.01))
    
        df['loyalty_index'] = 0
        for i in range(len(df['growth_past'])):
            if((df['market_cap'][i]>0.02) & (df['order_size'][i]>0.02)):
                df['loyalty_index'][i] = 1
        
        
        df['min_order_size_for_discount'] = [np.amin(df.loc[(df['Product Set'] == df['Product Set'][i]) & (df['Discount_Total']>0) & (df['order_size']>0) , "order_size"]) for i in range(len(df['growth_past']))]
    
        df['inventory_lingering_factor'] = [(np.amax(df.loc[df['Product Set'] == df['Product Set'][i] , "Discount_Total"])/(np.amax(df['Discount_Total'])+0.01)) * ((df['order_size'][i] - df['min_order_size_for_discount'][i])/(df['order_size'][i] + 0.01))*100  for i in range(len(df['growth_past'])) ]

        df['inventory_lingering_factor'] = df['inventory_lingering_factor'].fillna(0)
        df['min_order_size_for_discount'] = df['min_order_size_for_discount'].fillna(0)
    
    
        df['profit_Product'] = [np.mean(df.loc[ df['Product Set']==df['Product Set'][i] , "Discount_Total"]) for i in range(len(df['growth_past']))]
        maxDiscount = np.amax(df['profit_Product'])
        df['profitability_indicator'] = [(df['profit_Product'][i]/maxDiscount)*100 for i in range(len(df['growth_past']))]
    
    
        GTO = [np.mean(df.loc[ df['Product Set']==df['Product Set'][i] , "GTO_2019"]) for i in range(len(df['growth_past']))]
        n = len(df['growth_past'])
        pqr = [0 for i in range(n)]
        xyz = [0 for i in range(n)]
        a = df['order_size']
        b = df['min_order_size_for_discount']
        c = df['profit_Product']

        d = df['Expected_GTO']
        e = df['GTO_2019']
        f = df['Volume_2019 Product']
        for i in range(len(df['growth_past'])):
        
    
            xyz[i] = c[i]*(e[i]/(GTO[i]+0.01))*(d[i]/(e[i]+0.01))*1.5
        df['upper_limit'] = xyz
    
        return df

df = pd.read_excel("data.xlsx")
obj = featureEngineering_TrainTest(df)
newData = obj.featureEngineering()
newData.to_excel("data2.xlsx",index = False)