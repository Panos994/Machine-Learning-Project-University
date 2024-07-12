import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Drop unnecessary columns
    df = df.drop(columns=['Month', 'Browser', 'OperatingSystems'])
    
    # Apply One-hot encoding to categorical variables
    df = pd.get_dummies(df, columns=['Region', 'TrafficType', 'VisitorType'])
    
    # Split the target variable 'Revenue' from the features
    y = df['Revenue']
    X = df.drop(columns=['Revenue'])
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Load data and prepare
data = pd.read_csv('project2_dataset.csv')  

X_train2, X_test2, y_train2, y_test2 = prepare_data(data, train_size=0.75, shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = prepare_data(data, train_size=0.7, shuffle=True, random_state=42)

# Perform MinMax scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Logistic Regression object
logistic_regression = LogisticRegression(penalty='none', max_iter=1000)

# Train the model
logistic_regression.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = logistic_regression.predict(X_train_scaled)
y_test_pred = logistic_regression.predict(X_test_scaled)

# Calculate accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Training set shape: {X_train2.shape}")
print(f"Test set shape: {X_test2.shape}")
print(f"Training labels shape: {y_train2.shape}")
print(f"Test labels shape: {y_test2.shape}")


 # Εμφάνιση του σχήματος των συνόλων δεδομένων
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

print(f"Scaled training set shape: {X_train_scaled.shape}")
print(f"Scaled test set shape: {X_test_scaled.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")


print(f"Training accuracy: {accuracy_train:.2f}")
print(f"Test accuracy: {accuracy_test:.2f}")

# Calculate and display confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()