import os 
import pandas as pd

results_dir = "preds/reg"
true_file_suffix = ".NS_p_true.csv"
predicted_file_suffix = ".NS.csv"
stock_data_file = 'stock_data.txt'
options_file = 'stock_options.txt'

# file = "C:\Users\ashde\Desktop\dev\mtp\git_wip\stock-price-pred\preds\reg\stacking_regressor_new2AXISBANK.NS_p_true.csv"
for file in os.listdir(results_dir):
    if file.endswith(true_file_suffix):
        continue
    if file.endswith(predicted_file_suffix):
        symbol = file[len('stacking_regressor_new2'):-len(predicted_file_suffix)]
        true_file = os.path.join(results_dir, file[:-len(predicted_file_suffix)] + true_file_suffix )
        true_df = pd.read_csv(true_file)
        true_data = list(true_df.iloc[:, 1])
        line = "true_data['{}'] = {};".format(symbol, true_data)

        with open(stock_data_file, 'a') as f:
            f.write(line)
            f.write('\n')      
        
        prediction_file = os.path.join(results_dir, file)
        pred_df = pd.read_csv(prediction_file)
        pred_data = list(pred_df.iloc[:, 1])
        line = "pred_data['{}'] = {};".format(symbol, pred_data)
        
        with open(stock_data_file, 'a') as f:
            f.write(line)
            f.write('\n')
            f.write('\n')

        line = '<option value="{}">{}</option>'.format(symbol, symbol)

        with open(options_file, 'a') as f:
            f.write(line)
            f.write('\n')

        

