import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Drop τα columns αυτά για απλούστευση
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    
    # μετατρέπω τις boolean τιμές σε αριθμητικές
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    # εφαρμογή  One-hot encoding στα κατηγορικά variables
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    
    # χωρίζουμε τη μεταβλητή στόχο Revenue απο τα χαρακτηριστικά
    y = df['Revenue']
    X = df.drop(columns=['Revenue'])
    
    # χωρίζω το dataset σε σύνολο εκπαίδευσης και δοκιμής
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# φορτώνω τ δεδομένα
data = pd.read_csv('project2_dataset.csv')  

# προετοιμάζω τα δεδομένα με διαφορετικά train/test splits
X_train2, X_test2, y_train2, y_test2 = prepare_data(data, train_size=0.75, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = prepare_data(data, train_size=0.7, shuffle=True, random_state=42)

# Kάνω MinMax scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Logistic Regression object
logistic_regression = LogisticRegression(penalty='none', max_iter=1000)

# κάνω train το μοντέλο
logistic_regression.fit(X_train_scaled, y_train)

#  predictions
y_train_pred = logistic_regression.predict(X_train_scaled)
y_test_pred = logistic_regression.predict(X_test_scaled)

# υπολογισμός ευστοχίας
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Training set shape: {X_train2.shape}")
print(f"Test set shape: {X_test2.shape}")
print(f"Training labels shape: {y_train2.shape}")
print(f"Test labels shape: {y_test2.shape} \n")

#  shapes of datasets
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}\n")



print(f"Training accuracy: {accuracy_train:.2f}")
print(f"Test accuracy: {accuracy_test:.2f}\n")

#  confusion matrices for training and test sets
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Training):")
print(conf_matrix_train)

print("\n")

print("Confusion Matrix (Test):")
print(conf_matrix_test)

# Plot confusion matrices
plt.figure(figsize=(12, 5))

# Plots confusion matrix for training set
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Training)')

# Plot confusion matrix for test set
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (Test)')

plt.tight_layout()
plt.show()
