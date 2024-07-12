import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Αφαίρεση των χαρακτηριστικών Month, Browser, OperatingSystems
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    
    # Εφαρμογή One-hot encoding στις μεταβλητές Region, TrafficType, VisitorType
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    
    # Χωρισμός της μεταβλητής στόχου (Revenue) από τις υπόλοιπες
    y = df['Revenue']
    X = df.drop(columns=['Revenue'])
    
    # Χωρισμός του συνόλου δεδομένων σε σύνολο εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Φόρτωση δεδομένων και προετοιμασία
data = pd.read_csv('project2_dataset.csv')  
X_train, X_test, y_train, y_test = prepare_data(data, train_size=0.7, shuffle=True, random_state=42)

# Γραμμική κανονικοποίηση
scaler = MinMaxScaler()

# Υπολογισμός των παραμέτρων κανονικοποίησης στο σύνολο εκπαίδευσης
X_train_scaled = scaler.fit_transform(X_train)

# Εφαρμογή της κανονικοποίησης στο σύνολο εκπαίδευσης και δοκιμής
X_test_scaled = scaler.transform(X_test)

# Δημιουργία αντικειμένου LogisticRegression
logistic_regression = LogisticRegression(penalty='none', max_iter=1000)

# Εκπαίδευση του μοντέλου στα κανονικοποιημένα δεδομένα
logistic_regression.fit(X_train_scaled, y_train)

# Προβλέψεις για το σύνολο εκπαίδευσης και το σύνολο δοκιμής
y_train_pred = logistic_regression.predict(X_train_scaled)
y_test_pred = logistic_regression.predict(X_test_scaled)

# Εκτύπωση του αριθμού των επαναλήψεων που πήρε ο αλγόριθμος για να συγκλίνει
print("Number of iterations taken by the algorithm to converge:", logistic_regression.n_iter_)

# Εκτύπωση του ποσοστού ακρίβειας στο σύνολο εκπαίδευσης και το σύνολο δοκιμής
accuracy_train = (y_train == y_train_pred).mean()
accuracy_test = (y_test == y_test_pred).mean()
print(f"Training accuracy: {accuracy_train:.2f}")
print(f"Test accuracy: {accuracy_test:.2f}")
