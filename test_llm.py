import os
from google import genai

test_key = os.environ.get("GEMINI_API_KEY")
if not test_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

print(f"1. Attempting to initialize client with key: {test_key[:8]}...")

try:
    client = genai.Client(api_key=test_key)
    print("2. Client initialized. Attempting network call to Gemini 1.5 Flash...")
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Reply with the exact phrase: "LLM is working!"'
    )
    print("\n✅ SUCCESS! Response from Google:")
    print(response.text)
    
except Exception as e:
    print("\n❌ LLM CALL FAILED. Root Cause Error:")
    print(str(e))
