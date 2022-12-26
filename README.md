# sg-crime-prediction
Using Python and machine learning to build a predictive model for crime patterns using data from Singapore Open Data

# get_crime_data.py
GET request to the Singapore Open Data API to retrieve the crime data, converts the response to a JSON object, extracts the records from the JSON object, converts the records to a Pandas DataFrame, selects the features (location, type, and time) that will be used to predict the target (crime rate), encodes the categorical features using one-hot encoding, splits the data into training and testing sets, trains a random forest classifier on the training data, makes predictions on the testing data, and calculates the accuracy of the predictions.
