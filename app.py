
import pandas as pd
from flask import Flask, render_template, url_for,request
from predict import predictDiscounts
app = Flask(__name__)
import os
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
    

    trainData = pd.read_excel('data2.xlsx')
    if os.path.exists("static/images/poc.png"):
        os.remove("static/images/poc.png")
    else:
        print("The file does not exist")
    predictedDiscountsObj = predictDiscounts(trainData , data)
    predictedDiscount , predictedOnInvoiceDiscount , predictedOffInvoiceDiscount , randNumber = predictedDiscountsObj.predict()

    return render_template('predict.html',pred1='Total Discount = {}'.format(predictedDiscount) , pred2 = 'OnInvoice Discount = {}'.format(predictedOnInvoiceDiscount) , pred3 = 'OffInvoice Discount = {}'.format(predictedOffInvoiceDiscount) , path = "static/images/poc" + str(randNumber) + ".png" )


if __name__ == '__main__':
    app.run(host='0.0.0.0')