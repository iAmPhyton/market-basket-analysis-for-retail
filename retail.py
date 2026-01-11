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

#converting frozensets to strings for better readability
#converting apriori to plain tex: "Item"
rules['ant_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['con_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

#visuals: scatter plots of all rules
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="support",
    y="confidence",
    size="lift",
    hue="lift",    data=rules,
    palette="viridis",
    sizes=(20, 200)
)

plt.title('Market Basket Analysis: Support vs Confidence (Color =  Lift)')
plt.xlabel('Support (Popularity)')
plt.ylabel('Confidence (Likelihood)')
plt.legend(bbox_to_anchor=(1.01, 1),
           loc='upper left')
plt.show()

#visuals: Heatmap of the Top 15 Strongest Rules
#pivoting to data to create a matrix
#rows = Fist Item (Anecedent), Cols = Second Item (Consequent), Value = Lift
top_rules = rules.sort_values(by='lift', ascending=False).head(15)

#creating a matrix for the heatmap
pivot = top_rules.pivot(index='ant_str', columns='con_str', values='lift')

#visuals 
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True,
            cmap="Reds",
            fmt=".1f")
plt.title('Top 15 Association rules (Heatmap by Lift)')
plt.xlabel('Consequent (Buy This Next)')
plt.ylabel('Antecedent (If you buy this now..)')
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show() 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#using one of the most popular items: 'rabbit Night Light'
target_item = 'RABBIT NIGHT LIGHT'

#creating customer level dataset
#grouping by CustomerID
customer_data = france_subset.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique', #how many visits (frequency)
    'Quantity': 'sum', #total items bought (volume)
    'Description': 'nunique' #how many distinct products (variety)
}).rename(columns={
    'InvoiceNo': 'Frequency',
    'Quantity': 'TotalVolume',
    'Description': 'Variety'
})

#creating target column (did they buy rhe rabbit)
rabbit_buyers = france_subset[france_subset['Description'] == target_item]['CustomerID'].unique()

#creating new column: 'Target', 1 if they are in the rabbit_buyers list, 0 if not
customer_data['Target'] = customer_data.index.isin(rabbit_buyers).astype(int)
print(f"Dataset Shape: {customer_data.shape}")
print(customer_data.head())
print("\nTarget Distribution:")
print(customer_data['Target'].value_counts()) 

#training the Prophet
#spliting data
x = customer_data[['Frequency', 'TotalVolume', 'Variety']]
y = customer_data['Target']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42) 

#training the Prophet
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

#prediction and evaluation
y_pred = rf_model.predict(x_test)
print("--- Propensity Model Performance ---")
print(classification_report(y_test, y_pred)) 

#Feature Importance
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- What drives the purchase? ---")
print(importance) 

#creating a prediction for a fake customer (potential prediction)
def predict_rabbit_potential(frequency, volume, variety):
    #creating a dataframe for the new user data
    new_retail = pd.DataFrame([[frequency, volume, variety]], 
                            columns=['Frequency', 'TotalVolume', 'Variety'])
    
    #prediction (0 or 1)
    prediction = rf_model.predict(new_retail)[0]
    probability = rf_model.predict_proba(new_retail)[0][1] # Probability of being class 1
    
    if prediction == 1:
        return f"High Potential! (Probability: {probability:.2%}). Send them an ad!"
    else:
        return f"Low Potential (Probability: {probability:.2%}). Save the ad money!"

#testing on fake customers
#customer A: Shopped once, bought 5 items (Newbie)
print(f"Customer A: {predict_rabbit_potential(1, 5, 5)}")

#customer B: Shopped 10 times, bought 500 items (Power User)
print(f"Customer B: {predict_rabbit_potential(10, 500, 120)}")

#creating visuals: feature importance and confusion matrix
#feature importance visuals
plt.figure(figsize=(8,5))
#using the 'importance' dataframe
sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
plt.title('What Drives a "Rabbit Night Light" Purchase')
plt.xlabel('Importance Score')
plt.ylabel('Customer Behavior')
plt.show()

from sklearn.metrics import confusion_matrix
#confusion matrix visuals
#calculating the matrix
cm = confusion_matrix(y_test, y_pred)

#plots
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title('Propensity Model Performance')
plt.xlabel('Model Prediction')
plt.ylabel('Truth')
plt.show()

