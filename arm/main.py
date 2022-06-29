import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot as plt
import arm_encoder
import os
import pickle


encode = False
ohlcv_dir = '../data_collection/ohlcv_data'
ohlcv_prefix = 'ohlcv_'
ohlcv_suffix = '.csv'
save_dir = 'encoded'
save_prefix = 'arm_encoded'
save_suffix = '.csv'
fi_dir = 'frequent_itemsets'
ap_dir = 'ap_rules'
encoded_csv_path = None
min_support_freq_itemset = 0.03
confidence_threshold = 0.9


if __name__ == "__main__":
    if encode:
        df = arm_encoder.arm_encoder(ohlcv_dir, save_dir=save_dir, save_prefix=save_prefix,
        save_suffix=save_suffix, ohlcv_prefix=ohlcv_prefix, ohlcv_suffix=ohlcv_suffix, verbose=True)
    
    for file in os.listdir(save_dir):
        if file.startswith(save_prefix) and file.endswith(save_suffix):
            sector = file[len(save_prefix):-len(save_suffix)]
            df = pd.read_csv(os.path.join(save_dir, file))

<<<<<<< HEAD
            if 'Unnamed: 0' in df.columns:
                df.drop('Unnamed: 0', axis=1, inplace=True)
            if 'Date' in df.columns:
                df.drop('Date', axis=1, inplace=True)
            df.fillna(False, inplace=True)
            
            # Find and Save Frequent Itemsets. Don't run if already saved.
            frequent_itemsets_ap = apriori(df, min_support=0.03, use_colnames=True)
            fi_path = os.path.join(fi_dir, '{}_frequent_with_support_{}perc.pickle'.format(sector, int(100*min_support_freq_itemset)))
            with open(fi_path, 'wb') as handle:
                pickle.dump(frequent_itemsets_ap, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Load saved frequent itemsets and mine association rules.
            # Don't run if already mined.
            with open(fi_path, 'rb') as handle:
                frequent_itemsets_ap = pickle.load(handle)
            # frequent_itemsets_ap = pd.read_csv(fi_path)
            rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=confidence_threshold)
            
            rules_ap.sort_values(by='confidence', ascending=False, inplace=True)
            ap_path = os.path.join(ap_dir, '{}_ar_with_conf_{}perc.csv'.format(sector, int(100*confidence_threshold)))
            rules_ap.to_csv(ap_path)
=======
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Date' in df.columns:
        df.drop('Date', axis=1, inplace=True)
    df.fillna(value=False, inplace=True)
    
    
    # Find and Save Frequent Itemsets. Don't run if already saved.
    frequent_itemsets_ap = apriori(df, min_support=0.2, use_colnames=True)
    frequent_itemsets_ap.to_csv('frequent_with_support_2perc.csv')
    
    # Load saved frequent itemsets and mine association rules.
    # Don't run if already mined.
    frequent_itemsets_ap = pd.read_csv('frequent_with_support_2perc.csv')
    rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.90)
    
    rules_ap.sort_values(by='confidence', ascending=False, inplace=True)
    rules_ap.to_csv('ar_with_conf_90perc.csv')
>>>>>>> e6f155fcfbdf930542a34534c6c0b4db2fcefbf7
