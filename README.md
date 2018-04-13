# Matching challenge
## Data Challenge at Columbia, in March 2018. 
- **Goal** :  From two datasets that describe the same entities, identify which entity in one dataset is the same as an entity in the other dataset. Our datasets were provided by Foursquare and Locu, and contain descriptive information about various venues such as venue names and phone numbers.
Write a script that will load both datasets and identify matching venues in each dataset. It should generate **matches_test.csv**, a mapping that looks like **matches_train.csv**. F1-score is the metric used in this challenge (strongly imbalanced problem).
- Repository contains: 
 - The data and matches for training: 
  - foursquare_train.json
  - locu_train.json
  - matches_train.csv
 - The data for testing: 
  - foursquare_test.json
  - locu_test.json
 - .py file generating matches:
  - Entity_Resolution.py 
 - A notebook summarizing preprocessing and model tuning:
  - matching_notebook.ipynb
 - A report that gathers insights collected along the way: 
  - MatchingReport
- With a proper preprocessing and hyper-parameter tuning, a precision of 100%, a recall of 96.67% and a F1 score of 98.31% can be reached.
