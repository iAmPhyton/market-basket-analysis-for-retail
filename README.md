Market Basket Analysis (E-commerce)

Project Overview
- "Frequently Bought Together" recommendations drive significant revenue for online retailers (e.g., Amazon). This project uses Unsupervised Learning (Association Rule Mining) to uncover hidden purchasing patterns in a large e-commerce dataset.

The goal of the project was to know if a customer buys Item A, how likely are they to buy Item B?

Dataset
- Source: [Online Retail Dataset (UCI ML Repo)](https://archive.ics.uci.edu/ml/datasets/online+retail)
- Scope: Focused on the "France" market segment (~8,300 transactions) to analyze specific regional buying habits.
- Structure: Transactional data (InvoiceNo, StockCode, Description, Quantity).

Challenges & Solutions
- The Problem:
When visualizing the Association Rules using a Heatmap, a `ValueError: Index contains duplicate entries` occurred. This happened because converting `frozenset({'Item A', 'Item B'})` to a string by simply taking the first item (`list(x)[0]`) created duplicate labels for different rules.

The Solution:
- Refactored the string conversion logic to join *all* items in the set: `', '.join(list(x))`.
- This ensured every rule had a unique string identifier, allowing the Pivot Table to reshape correctly.

Part 2: Propensity Modeling (Supervised Learning):
- Goal: Move beyond "rules" and predict *specific* customers likely to buy the popular "Rabbit Night Light."

Approach:
1.  Feature Engineering: Aggregated transaction logs into a Customer-Level dataset.
    - Features: Frequency (Visits), TotalVolume (Items Bought), Variety (Unique Items).
    - Target: Did they buy the Rabbit Night Light? (1/0).
2.  Modeling: Trained a Random Forest Classifier to distinguish buyers from non-buyers.

Results:
- Key Driver: `TotalVolume` was the #1 predictor of purchase likelihood.
- Application: Created a `predict_rabbit_potential()` function to score new users in real-time for targeted ad campaigns.

Contact
- Name: Chukwuemeka Eugene Obiyo
- Email: praise609@gmail.com
- Linkedin: https://www.linkedin.com/in/chukwuemekao/
