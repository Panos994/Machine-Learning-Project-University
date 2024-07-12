# linear_classification.py q3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

# Προετοιμασία δεδομένων και κανονικοποίηση
if __name__ == "__main__":
    # Φόρτωση δεδομένων
    data = pd.read_csv('project2_dataset.csv')  # Αντικαταστήστε το με το σωστό όνομα του αρχείου σας
    
    # Κλήση της συνάρτησης prepare_data με 70%-30% χωρισμό και σπόρο 42
    X_train, X_test, y_train, y_test = prepare_data(data, train_size=0.7, shuffle=True, random_state=42)
    
    # Γραμμική κανονικοποίηση
    scaler = MinMaxScaler()
    
    # Υπολογισμός των παραμέτρων κανονικοποίησης στο σύνολο εκπαίδευσης
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Εφαρμογή της κανονικοποίησης στο σύνολο εκπαίδευσης και δοκιμής
    X_test_scaled = scaler.transform(X_test)
    
    # Εμφάνιση του σχήματος των κανονικοποιημένων συνόλων δεδομένων
    print(f"Scaled training set shape: {X_train_scaled.shape}")
    print(f"Scaled test set shape: {X_test_scaled.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
