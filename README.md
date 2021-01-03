# Harvard CS182 (Artifical Intelligence) Psets - Developing an Ethical Credit Risk Model  :robot:
A repo with code for the final project of Harvard's CS182 (Artificial Intelligence) using artificial intelligence to predict if low-income households are at risk of meeting loan repayments, a collaborative project with [Cole Bateman](https://www.linkedin.com/in/cole-bateman/).

* **CS182 Paper** contains the written element of the project, entitled: *"Lost interest? Using artificial intelligence to predict if low-income households are at risk of meeting loan repayments"*. *(Grade: 50/55)*  :bank:
* **Code** contains the code to go along with the report.  :computer:



**Code Setup:**

1. Run 'bash make_data_directory_structure,sh' from the top directory
2. Get data sets 'application_test.csv', 'application_train.csv', 'bureau.csv', and 
   'bureau_balance.csv' from https://www.kaggle.com/c/home-credit-default-risk/data
3. Insert data sets in data/raw
4. Run python files in alphabetical order, starting with 'a_data_manipulation.py' and 
   ending with 'e_lstm_plus_static_keras.py'
		-Note, you can run the baseline 'random_forest.py' and any point in the process 
		 to get comparative data
		-Note, 'd_join_sequence_and_features.py' takes the longest to run.

(Ignore the file 'scrap_code.py', it was used for testing and storing unused code)
