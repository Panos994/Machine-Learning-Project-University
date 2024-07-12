# eda.py - Ερώτημα 1ο 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# φορτώνω τα  δεδομένα με το csv που δοθηκε για την εργασία μας 
data = pd.read_csv('project2_dataset.csv')  



# Ερώτηση 1: Πόσες είναι οι εγγραφές του συνόλου δεδομένων;
num_records = data.shape[0]
print(f'Number of records: {num_records}')

# Ερώτηση 2: Σε τι ποσοστό από αυτές οι χρήστες αγόρασαν τελικά;
purchase_rate = data['Revenue'].mean() * 100
print(f'Purchase rate: {purchase_rate:.2f}%')

# Ερώτηση 3: Ποια είναι η ευστοχία (accuracy) ενός μοντέλου το οποίο προβλέπει πάντα ότι ο χρήστης δε θα αγοράσει;
accuracy_non_purchase_model = (data['Revenue'].value_counts(normalize=True)[0]) * 100
print(f'Accuracy of non-purchase model: {accuracy_non_purchase_model:.2f}%')

# Προαιρετικά εκδίδω και κάποια διαγράμματα

# Κατανομή της μεταβλητής στόχου (Revenue)
sns.countplot(x='Revenue', data=data)
plt.title('Distribution of Revenue')
plt.show()

# σχετίζω εδώ τα  χαρακτηριστικά με τη μεταβλητή στόχο (Revenue)
for column in data.columns:
    if column != 'Revenue':
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='Revenue', y=column, data=data)
        plt.title(f'Relationship between {column} and Revenue')
        plt.show()

