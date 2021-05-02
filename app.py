
import pandas as pd
from flask import Flask, render_template, url_for,request
from predict import predictDiscounts
import logging
import sys
import os
import matplotlib.pyplot as plt
import waterfall_chart
import base64
import io

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)





@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    keys = [x for x in request.form.keys()]
    print(int_features)
    print(keys)
    columns = keys
    data = dict()
    for i in range(len(int_features)):
        if(columns[i] == 'Volume_2019' or columns[i] == 'Volume_2018' or columns[i] == 'GTO_2019' or columns[i] == 'Volume_2019 Product' or columns[i] == 'Tax'):
            data[columns[i]] = float(int_features[i])
        else:
            data[columns[i]] = int_features[i]

    data['Product Set'] = data['Returnalility'] + '_' + data['Pack_Type'] + '_' + data['Brand'] + '_' + data['Sub-Brand']
    

    trainData = pd.read_csv('data3.csv')
    if os.path.exists("static/images/poc.png"):
        os.remove("static/images/poc.png")
    else:
        print("The file does not exist")
    predictedDiscountsObj = predictDiscounts(trainData , data)
    predictedDiscount , predictedOnInvoiceDiscount , predictedOffInvoiceDiscount , randNumber , index , values = predictedDiscountsObj.predict()
    print(index)
    print(values)
    figure_size = (14,14)
    fig = Figure(figsize = figure_size)
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Profitability Chart of POC")
    axis.set_xlabel("Parameters")
    axis.set_ylabel("Earnings Values")
    axis.grid()
    axis.bar(index, values ,width = 0.6)
    axis.set_xticklabels(index, rotation = 45)
    
    for i in axis.patches:
        
        axis.text(i.get_x(), i.get_height(), str(round(i.get_height(), 2)), fontsize=16, color='red') 
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    
    
    
    
        
        # Encode PNG image to base64 string
    
        
       


    return render_template('predict.html',pred1='Total Discount = {}'.format(predictedDiscount) , pred2 = 'OnInvoice Discount = {}'.format(predictedOnInvoiceDiscount) , pred3 = 'OffInvoice Discount = {}'.format(predictedOffInvoiceDiscount) , image=pngImageB64String )







if __name__ == '__main__':
    app.run(debug = True)
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)