import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class CrimePredictor:
    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.clf = RandomForestClassifier()
    
    def retrieve_data(self):
        # Make a GET request to the Singapore Open Data API to retrieve the crime data
        response = requests.get(f"https://data.gov.sg/api/action/datastore_search?resource_id={self.resource_id}")

        # Convert the response to a JSON object
        data = response.json()

        # Extract the records from the JSON object
        records = data['result']['records']

        # Convert the records to a Pandas DataFrame
        self.df = pd.DataFrame.from_records(records)

    def build_model(self):
        # Select the features (columns) that will be used to predict the target (crime rate)
        features = ['location', 'type', 'time']
        X = self.df[features]

        # Encode the categorical features using one-hot encoding
        X = pd.get_dummies(X)

        # Select the target (crime rate)
        y = self.df['crime_rate']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the random forest classifier on the training data
        self.clf.fit(X_train, y_train)

    def predict(self, data):
        # Make predictions on the given data
        predictions = self.clf.predict(data)
        return predictions

    def evaluate(self):
        # Select the features (columns) that will be used to predict the target (crime rate)
        features = ['location', 'type', 'time']
        X = self.df[features]

        # Encode the categorical features using one-hot encoding
        X = pd.get_dummies(X)

        # Select the target (crime rate)
        y = self.df['crime_rate']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Make predictions on the testing data
        predictions = self.clf.predict(X_test)

        # Calculate the accuracy of the predictions
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.2f}")

# Create a CrimePredictor object with the appropriate resource ID
predictor = CrimePredictor(resource_id='xxx-xxxx-xxxx-xxxx')

# Retrieve the data from the API
predictor.retrieve_data()

# Build the model using the data
predictor.build_model()

# Evaluate the model
predict
