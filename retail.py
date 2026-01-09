import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#loading dataset
retail = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
print(retail.tail())

#basic cleaning
#dropping rows with missing customerID
retail.dropna(subset=['CustomerID'], inplace=True)
#removing cancelled transactions
retail = retail[~retail['InvoiceNo'].astype(str).str.startswith('C')]
#removing extra spaces from descriptions
retail['Description'] = retail['Description'].str.strip()
#filter for france only (its a very large dataset..)
france_subset = retail[retail['Country'] == "France"]
print(f"Total Transactions: {retail.shape[0]}")
print(f"France Transactions: {france_subset.shape[0]}")
print(france_subset.head()) 

#creating basket matrix
basket = (france_subset.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#conerting to binary
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.map(encode_units)
basket_sets = basket.applymap(encode_units) 

#dropping the 'Postage' column as it is usually a line item, not a real product
if 'POSTAGE' in france_subset.columns:
    basket_sets.drop('POSTAGE', inplace=True, axis=1)
print((f"Matrix Shape: {basket_sets.shape}"))
print(basket_sets.head()) 

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
#finding frequent itemsets
#min_support=0.07 means items that appear in at least 7% of txns
#use_colnames=True gives actual product names
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

#sorting by support (most popular items first)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(f"Found {frequent_itemsets.shape[0]} frequent itemsets.")
print(frequent_itemsets.sort_values(by='support', ascending=False).head()) 

#generating rules (confidence and lift)
#metric="lift": rules where the link is stronger than random chance
#min_threshold=1: keeping positive associations
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1) 

#sorting by the strongest connection (lift)
rules =  rules.sort_values(by='lift', ascending=False)

#cleaning up display by showing only key columns 
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)) 

