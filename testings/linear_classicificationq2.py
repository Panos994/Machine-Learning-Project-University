# linear_classification.py q2

import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Αφαίρεση των χαρακτηριστικών Month, Browser, OperatingSystems
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    
    # Μετατροπή boolean τιμών σε αριθμητικές
    df['Weekend'] = df['Weekend'].apply(lambda x: 1 if x == 'TRUE' else 0)
    df['Revenue'] = df['Revenue'].apply(lambda x: 1 if x == 'TRUE' else 0)
    
    # Εφαρμογή One-hot encoding στις μεταβλητές Region, TrafficType, VisitorType
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    
    # Χωρισμός της μεταβλητής στόχου (Revenue) από τις υπόλοιπες
    X = df.drop(columns=['Revenue'])
    y = df['Revenue']

    
    # Χωρισμός του συνόλου δεδομένων σε σύνολο εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Παράδειγμα χρήσης της συνάρτησης prepare_data
if __name__ == "__main__":
    # Φόρτωση δεδομένων
    data = pd.read_csv('project2_dataset.csv')  
    
    #Κλήση της συνάρτησης prepare_data
    X_train, X_test, y_train, y_test = prepare_data(data, train_size=0.75, shuffle=True, random_state=42)
    
    # Εμφάνιση του σχήματος των συνόλων δεδομένων
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
