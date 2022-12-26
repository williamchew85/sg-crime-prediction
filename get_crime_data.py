import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Make a GET request to the Singapore Open Data API to retrieve the crime data
response = requests.get("https://data.gov.sg/api/action/datastore_search?resource_id=xxx-xxxx-xxxx-xxxx")

# Convert the response to a JSON object
data = response.json()

# Extract the records from the JSON object
records = data['result']['records']

# Convert the records to a Pandas DataFrame
df = pd.DataFrame.from_records(records)

# Select the features (columns) that will be used to predict the target (crime rate)
features = ['location', 'type', 'time']
X = df[features]

# Encode the categorical features using one-hot encoding
X = pd.get_dummies(X)

# Select the target (crime rate)
y = df['crime_rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions on the testing data
predictions = clf.predict(X_test)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
