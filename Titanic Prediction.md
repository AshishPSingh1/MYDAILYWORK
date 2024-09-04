!pip install pandas numpy scikit-learn matplotlib seaborn


from google.colab import files

uploaded = files.upload()


import pandas as pd

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Display the first few rows of the dataset
print(train_df.head())


print(train_df.info())
print(train_df.describe())
print(train_df['Survived'].value_counts())

# Fill missing Age values with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop the Cabin column as it has too many missing values
train_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
test_df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)


# Convert 'Sex' into numerical feature
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' into numerical feature
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Select features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]
y = train_df['Survived']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_val_scaled)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_val, y_pred)}')
print(classification_report(y_val, y_pred))


# Prepare the test data
X_test = test_df[features]

# Impute missing values in X_test (replace NaN with median for numerical features)
X_test['Age'].fillna(X_test['Age'].median(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].median(), inplace=True) 

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Predict survival on test data
test_predictions = model.predict(X_test_scaled)

# ... rest of your code ...

# Save predictions to a CSV file
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_predictions})
submission_df.to_csv('gender_submission.csv', index=False)

# Download the submission file
files.download('gender_submission.csv')
