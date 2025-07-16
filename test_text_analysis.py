import requests
import json

# The URL of your locally running FastAPI application
API_URL = "http://localhost:8000/analyze_text"

# Sample product data. The description contains keywords often found in fake listings.
test_product = {
    "name": "100% Genuine Amazing Leather Wallet - Best Quality Original",
    "description": (
        "Experience the pinnacle of craftsmanship with our 100% authentic leather wallet. "
        "This amazing, perfect wallet is an original design, guaranteed to be the best you have ever owned. "
        "It features advanced RFID blocking to protect your cards. Free shipping on all orders! "
        "Click here to buy now: www.not-a-real-site.com"
    )
}

def run_test():
    """
    Sends a test request to the /analyze_text endpoint and prints the response.
    """
    print("Sending test request to:", API_URL)
    print("Product Data:")
    print(json.dumps(test_product, indent=2))
    
    try:
        response = requests.post(API_URL, json=test_product)
        
        # Raise an exception if the request returned an error code
        response.raise_for_status()
        
        print("\n✅ Test Successful!")
        print("Response from server:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Test Failed: Could not connect to the server.")
        print(f"Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_test() 