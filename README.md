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

Contact
- Name: Chukwuemeka Eugene Obiyo
- Linkedin: https://www.linkedin.com/in/chukwuemekao/
