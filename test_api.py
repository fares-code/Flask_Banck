import requests
import json

# Test data
test_data = {
    "age": 30,
    "duration": 300,
    "campaign": 2,
    "previous": 1
}

# Send POST request to your API
response = requests.post(
    "http://localhost:5000/predict",
    headers={"Content-Type": "application/json"},
    data=json.dumps(test_data)
)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())