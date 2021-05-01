import pandas as pd
import numpy as np
import joblib
import numpy
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import waterfall_chart
import random


class Features:
    def __init__(self,data , dictOfInputs):
        self.data = data
        self.dictOfInputs = dictOfInputs
    
    def getNewFeatures(self):
        df = self.data
        dictOfOutputs = self.dictOfInputs
        epsilon = 0.01
        dictOfInputs = self.dictOfInputs
        dictOfOutputs['growth_past'] = dictOfInputs['Volume_2019'] - dictOfInputs['Volume_2018']
        dictOfOutputs['industry_growth'] = np.mean(df.loc[(df['segment']==dictOfInputs['segment'])  & (df['poc_image']==dictOfInputs['poc_image']) & (df['sdfc_Tier'] == dictOfInputs['sdfc_Tier']) & (df['sub_segment']==dictOfInputs['sub_segment']) & (df['province']==dictOfInputs['province']) , "growth_past" ])
        dictOfOutputs['market_cap'] = dictOfInputs['Volume_2019'] / (np.sum(df.loc[(df['segment']==dictOfInputs['segment'])  & (df['poc_image']==dictOfInputs['poc_image']) & (df['sdfc_Tier'] == dictOfInputs['sdfc_Tier']) & (df['sub_segment']==dictOfInputs['sub_segment']) & (df['province']==dictOfInputs['province']) , "Volume_2019" ]) + epsilon)
        dictOfOutputs['future_growth'] = (dictOfOutputs['growth_past'] + dictOfOutputs['market_cap'] * dictOfOutputs['industry_growth'])/(1+dictOfOutputs['market_cap'])
        dictOfOutputs['order_size'] = dictOfOutputs['Volume_2019 Product']/ (np.amax(df.loc[df['Product Set'] == dictOfOutputs['Product Set'] , "Volume_2019 Product"]))
        dictOfOutputs['Expected_GTO'] = dictOfOutputs['GTO_2019'] * ((dictOfOutputs['Volume_2019'] + dictOfOutputs['future_growth'])/(dictOfOutputs['Volume_2019'] + epsilon))
        dictOfOutputs['Expected_product_volume'] = dictOfOutputs['Volume_2019 Product'] * ((dictOfOutputs['Volume_2019'] + dictOfOutputs['future_growth'])/(dictOfOutputs['Volume_2019'] + epsilon))
        dictOfOutputs['loyalty_index'] = 0
        if((dictOfOutputs['market_cap']>0.02) & (dictOfOutputs['order_size']>0.02)):
            dictOfOutputs['loyalty_index'] = 1
        dictOfOutputs['min_order_size_for_discount'] = np.amin(df.loc[ (df['Product Set'] == dictOfOutputs['Product Set']) & (df['Discount_Total']>0) & (df['order_size']>0) , "order_size"])
        dictOfOutputs['inventory_lingering_factor'] = (np.amax(df.loc[df['Product Set'] == dictOfOutputs['Product Set'] , "Discount_Total"]) / (np.amax(df['Discount_Total']) + epsilon)) * ((dictOfOutputs['order_size'] - dictOfOutputs['min_order_size_for_discount']) / (dictOfOutputs['order_size'] + epsilon)) * 100
        dictOfOutputs['profit_Product'] = np.mean(df.loc[ df['Product Set'] == dictOfOutputs['Product Set'] , "Discount_Total"])
        maxDiscount = np.max(df['profit_Product'])
        dictOfOutputs['profitability_indicator'] = (dictOfOutputs['profit_Product']/(maxDiscount+epsilon))*100
        a = dictOfOutputs['order_size']
        b = dictOfOutputs['min_order_size_for_discount']
        c = dictOfOutputs['profit_Product']

        d = dictOfOutputs['Expected_GTO']
        e = dictOfOutputs['GTO_2019']
        f = dictOfOutputs['Volume_2019 Product']
        GTO = np.mean(df.loc[df['Product Set'] == dictOfOutputs['Product Set'] , "GTO_2019"])
        dictOfOutputs['ProfitMargin'] = (dictOfOutputs['profit_Product']/GTO)*1.5
        dictOfOutputs['upper_limit'] = c*(e/(GTO+0.01))*(d/(e+0.01))*1.2
        return dictOfOutputs

class encodeCategoricalVars:
    def __init__(self,data):
        self.data = data
    def encode(self):
        lb_make = LabelEncoder()
        lb_make.classes_ = numpy.load('classes_sdfc_tier.npy' , allow_pickle = True)
        print(lb_make.classes_)
        self.data['sdfc_Tier'] = lb_make.transform(self.data['sdfc_Tier'])
        
        for i in range(len(self.data['GTO_2019'])):
            if(self.data['poc_image'][i]==0):
                self.data['poc_image'][i] = "Mainstream"
        lb_make.classes_ = numpy.load('classes_poc_image.npy' , allow_pickle = True) 
        self.data['poc_image'] = lb_make.transform(self.data['poc_image'])
        lb_make.classes_ = numpy.load('classes_segment.npy' , allow_pickle = True)
        self.data['segment'] = lb_make.transform(self.data['segment'])
        lb_make.classes_ = numpy.load('classes_sub_segment.npy' , allow_pickle = True)
        self.data['sub_segment'] = lb_make.transform(self.data['sub_segment'])
        lb_make.classes_ = numpy.load('classes_Product Set.npy' , allow_pickle = True)
        self.data['Product Set'] = lb_make.transform(self.data['Product Set'])
        lb_make.classes_ = numpy.load('classes_Brand.npy' , allow_pickle = True)
        self.data['Brand'] = lb_make.transform(self.data['Brand'])
        lb_make.classes_ = numpy.load('classes_Sub_Brand.npy' , allow_pickle = True)
        self.data['Sub-Brand'] = lb_make.transform(self.data['Sub-Brand'])
        lb_make.classes_ = numpy.load('classes_Pack_Type.npy' , allow_pickle = True)
        print(lb_make.classes_)
        self.data['Pack_Type'] = lb_make.transform(self.data['Pack_Type'])
        lb_make.classes_ = numpy.load('classes_Returnalility.npy' , allow_pickle = True)
        self.data['Returnalility'] = lb_make.transform(self.data['Returnalility'])
        lb_make.classes_ = numpy.load('classes_province.npy' , allow_pickle = True)
        print(lb_make.classes_)
        self.data['province'] = lb_make.transform(self.data['province'])
        
        return self.data
class plotSaver:
    def __init__(self,row_test,randNumber):
        self.row_test = row_test
        self.randNumber = randNumber

    def savePlot(self):
        row = self.row_test
        index = ['GTO_2019','Taxes Deducted','Operating Expenses' ,'Expected GTO' , 'Taxes' , 'Operating Expenses']
        values = list()
        values.append(row['GTO_2019'])
        values.append(-1*row['Tax'])
        values.append(-1*(1-row['ProfitMargin'])*(row['GTO_2019']-row['Tax']))
        values.append(row['Expected_GTO'])
        values.append(-1*(row['Tax'])*(row['Expected_GTO'] / row['GTO_2019']))
        values.append(-1*(1-row['ProfitMargin'])*(row['Expected_GTO']+values[-1]))

        figure = waterfall_chart.plot(index,values , rotation_value = 45)
        figure.title("Profit Estimation of Order")
        figure.savefig(r'static\images\poc'+str(self.randNumber) + '.png')
class predictDiscounts:
    def __init__(self,data,row_test):
        self.data = data
        self.row_test = row_test

    def predict(self):
        featuresObj = Features(self.data,self.row_test)
        self.row_test = featuresObj.getNewFeatures()
        row = self.row_test
        
        row = pd.DataFrame(row , index = [0])
        encodeObj = encodeCategoricalVars(row)
        row = encodeObj.encode()
        row = row.iloc[0]
        row = dict(row)
        print(row)
        filename1 = 'lowGTOModel_TotalDiscount.sav'
        lowGTOModel_TotalDiscount = joblib.load(open(filename1, 'rb'))

        filename2 = 'midGTOModel_TotalDiscount.sav'
        midGTOModel_TotalDiscount = joblib.load(open(filename2, 'rb'))

        filename3 = 'highGTOModel_TotalDiscount.sav'
        highGTOModel_TotalDiscount = joblib.load(open(filename3, 'rb'))

        filename4 = 'lowGTOModel_OnInvoiceDiscount.sav'
        lowGTOModel_OnInvoiceDiscount = joblib.load(open(filename4, 'rb'))

        filename5 = 'midGTOModel_OnInvoiceDiscount.sav'
        midGTOModel_OnInvoiceDiscount = joblib.load(open(filename5, 'rb'))

        filename6 = 'highGTOModel_OnInvoiceDiscount.sav'
        highGTOModel_OnInvoiceDiscount = joblib.load(open(filename6, 'rb'))

        if(row['GTO_2019']<10000):
            '''
            Use the lowGTO Models
            '''
            columnsToUse = ['Volume_2019' , 'Volume_2018'  , 'Expected_GTO'  , 'Expected_product_volume', 'profitability_indicator' , 'upper_limit'  ,'sdfc_Tier'  , 'loyalty_index' , 'Returnalility', 'market_cap' ]
            dataInputed = list()
            for i in columnsToUse:
                dataInputed.append(row[i])
            dataInputed = np.array(dataInputed)
            dataInputed = dataInputed.reshape(1,-1)
            predictedDiscount = lowGTOModel_TotalDiscount.predict(dataInputed)
            predictedOnInvoiceDiscount = lowGTOModel_OnInvoiceDiscount.predict(dataInputed)
            predictedOffInvoiceDiscount = predictedDiscount - predictedOnInvoiceDiscount
            if(predictedOnInvoiceDiscount>predictedDiscount and predictedDiscount>0):
                print("Adjustments Needed")
                predictedOnInvoiceDiscount = predictedDiscount
                predictedOffInvoiceDiscount = 0
            if(predictedOnInvoiceDiscount<predictedDiscount and predictedDiscount<0):
                print("Adjustments Needed")
                predictedOnInvoiceDiscount = predictedDiscount
                predictedOffInvoiceDiscount = 0
                
                
            print(predictedDiscount)
            print(predictedOnInvoiceDiscount)
            print(predictedOffInvoiceDiscount)
            randNumber = random.randint(1,10000000)
            plotObj = plotSaver(row,randNumber)
            plotObj.savePlot()


            return predictedDiscount , predictedOnInvoiceDiscount , predictedOffInvoiceDiscount , randNumber

        if((row['GTO_2019']>10000) and (row['GTO_2019']<50000)):
            '''
            Use the midGTO Models
            '''
            columnsToUse = ['Volume_2019' , 'Volume_2018' ,'Volume_2019 Product' ,'Expected_GTO','Expected_product_volume' , 'profitability_indicator' , 'upper_limit'  ,'sdfc_Tier'  , 'loyalty_index' , 'Returnalility',  'inventory_lingering_factor', 'market_cap', 'order_size']
            dataInputed = list()
            for i in columnsToUse:
                dataInputed.append(row[i])
            dataInputed = np.array(dataInputed)
            dataInputed = dataInputed.reshape(1,-1)
            predictedDiscount = midGTOModel_TotalDiscount.predict(dataInputed)
            predictedOnInvoiceDiscount = midGTOModel_OnInvoiceDiscount.predict(dataInputed)
            predictedOffInvoiceDiscount = predictedDiscount - predictedOnInvoiceDiscount
            if(predictedOnInvoiceDiscount>predictedDiscount and predictedDiscount>0):
                print("Adjustments Needed")
                predictedOnInvoiceDiscount = predictedDiscount
                predictedOffInvoiceDiscount = 0
            if(predictedOnInvoiceDiscount<predictedDiscount and predictedDiscount<0):
                print("Adjustments Needed")
                predictedOnInvoiceDiscount = predictedDiscount
                predictedOffInvoiceDiscount = 0
                
                
            print(predictedDiscount)
            print(predictedOnInvoiceDiscount)
            print(predictedOffInvoiceDiscount)
            randNumber = random.randint(1,10000000)
            plotObj = plotSaver(row,randNumber)
            plotObj.savePlot()


            return predictedDiscount , predictedOnInvoiceDiscount , predictedOffInvoiceDiscount , randNumber
        
        if(row['GTO_2019']>50000):
            '''
            Use the lowGTO Models
            '''
            columnsToUse = ['Volume_2019' , 'Volume_2018' ,'Volume_2019 Product' ,'Expected_GTO','Expected_product_volume' , 'profitability_indicator' , 'upper_limit'  ,  'inventory_lingering_factor', 'order_size']
            dataInputed = list()
            for i in columnsToUse:
                dataInputed.append(row[i])
            dataInputed = np.array(dataInputed)
            dataInputed = dataInputed.reshape(1,-1)
            predictedDiscount = highGTOModel_TotalDiscount.predict(dataInputed)
            predictedOnInvoiceDiscount = highGTOModel_OnInvoiceDiscount.predict(dataInputed)
            predictedOffInvoiceDiscount = predictedDiscount - predictedOnInvoiceDiscount
            if(predictedOnInvoiceDiscount>predictedDiscount and predictedDiscount>0):
                print("Adjustments Needed")
                predictedOnInvoiceDiscount = predictedDiscount
                predictedOffInvoiceDiscount = 0
            if(predictedOnInvoiceDiscount<predictedDiscount and predictedDiscount<0):
                print("Adjustments Needed")
                predictedOnInvoiceDiscount = predictedDiscount
                predictedOffInvoiceDiscount = 0
                
                
            print(predictedDiscount)
            print(predictedOnInvoiceDiscount)
            print(predictedOffInvoiceDiscount)
            randNumber = random.randint(1,10000000)
            plotObj = plotSaver(row,randNumber)
            plotObj.savePlot()


            return predictedDiscount , predictedOnInvoiceDiscount , predictedOffInvoiceDiscount , randNumber

