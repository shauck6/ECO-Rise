import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests
import json
from ibm_watson import DiscoveryV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Way to import data from a CSV file
def import_data_from_cos(api_key, endpoint_url, bucket_name, object_name):
    import ibm_boto3
    from ibm_botocore.client import Config

    cos = ibm_boto3.client(
        's3',
        ibm_api_key_id=api_key,
        ibm_service_instance_id=bucket_name,
        config=Config(signature_version='oauth'),
        endpoint_url=endpoint_url
    )

    file = cos.get_object(Bucket=bucket_name, Key=object_name)
    df = pd.read_csv(file['Body'])
    return dfnp.random.seed(42)
# Usage:
# api_key = "your-api-key"
# endpoint_url = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
# bucket_name = "your-bucket-name"
# object_name = "your-file.csv"
# df = import_data_from_cos(api_key, endpoint_url, bucket_name, object_name)

# After importing, you can use the dataframe in your model:
# X = df[['green_space_connectivity', 'wall_characteristics', 'sun_exposure', 'wind_speed', 'x_coordinate', 'y_coordinate', 'z_coordinate']]
# y = df['suitability_score']  # Assuming you have this column in your data


# Simulated data (replace this with actual data from Watson AI)
# n_samples = 1000

data = pd.DataFrame({
    'green_space_connectivity': np.random.rand(n_samples),
    'wall_characteristics': np.random.randint(1, 6, n_samples),  # 1-5 rating
    'sun_exposure': np.random.rand(n_samples),
    'wind_speed': np.random.rand(n_samples) * 10,  # 0-10 m/s
    'x_coordinate': np.random.rand(n_samples) * 100,  # 0-100 m
    'y_coordinate': np.random.rand(n_samples) * 100,  # 0-100 m
    'z_coordinate': np.random.rand(n_samples) * 30,   # 0-30 m (assuming a multi-story building)
})

# Calculate a "suitability score" (this is a simplification; adjust as needed)
data['suitability_score'] = (
    data['green_space_connectivity'] * 0.3 +
    data['wall_characteristics'] * 0.2 +
    data['sun_exposure'] * 0.3 +
    (1 - data['wind_speed'] / 10) * 0.2  # Assuming lower wind speed is better
)

# Prepare features and target
X = data[['green_space_connectivity', 'wall_characteristics', 'sun_exposure', 'wind_speed', 'x_coordinate', 'y_coordinate', 'z_coordinate']]
y = data['suitability_score']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"Model R-squared score on training data: {train_score:.4f}")
print(f"Model R-squared score on test data: {test_score:.4f}")

# Function to predict best location for green wall
def predict_best_location(model, scaler, n_predictions=5):
    # Create a grid of possible locations
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 100, 50)
    z = np.linspace(0, 30, 15)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    # Flatten the grid
    locations = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    
    # Assume average values for other features
    other_features = np.tile([0.5, 3, 0.5, 5], (locations.shape[0], 1))
    
    # Combine all features
    all_features = np.hstack((other_features, locations))
    
    # Scale the features
    all_features_scaled = scaler.transform(all_features)
    
    # Make predictions
    predictions = model.predict(all_features_scaled)
    
    # Get the top n predictions
    top_indices = predictions.argsort()[-n_predictions:][::-1]
    top_locations = locations[top_indices]
    top_scores = predictions[top_indices]
    
    return top_locations, top_scores

# Find the best locations
best_locations, best_scores = predict_best_location(model, scaler)

print("\nBest locations for green wall placement:")
for i, (location, score) in enumerate(zip(best_locations, best_scores), 1):
    print(f"{i}. Coordinates (x, y, z): ({location[0]:.2f}, {location[1]:.2f}, {location[2]:.2f}) m")
    print(f"   Predicted suitability score: {score:.4f}")
    print()

# Print feature importances
feature_names = ['Green Space Connectivity', 'Wall Characteristics', 'Sun Exposure', 'Wind Speed', 'X Coordinate', 'Y Coordinate', 'Z Coordinate']
importances = model.coef_
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
