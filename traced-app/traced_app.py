
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

print("BASE URL:", os.getenv("openai_MODEL_NAME"))
base_url = os.getenv("openai_BASE_URL")
model_name = os.getenv("openai_MODEL_NAME")
token = os.getenv("openai_APIKEY")

print("baseurl",base_url)
print("modelname",model_name)
print("token",token)
# Validate credentials
if not all([base_url, model_name, token]):
    raise ValueError(
        "Missing IBM watsonx environment variables. "
        "Check your .env file."
    )

llm = ChatOpenAI(
    model=os.getenv("openai_MODEL_NAME"),
    base_url=os.getenv("openai_BASE_URL"),
    api_key=os.getenv("openai_APIKEY"),
    temperature=0
)

# Test with three prompts
prompts = [
    "Explain LangChain in simple terms.",
    "What is a Vector Store?",
    "Explain the difference between RAG and AI Agents."
]

for i, prompt in enumerate(prompts, start=1):
    print(f"\n--- Prompt {i} ---")
    print(prompt)

    response = llm.invoke(prompt)

    print("\nResponse:")
    print(response.content)
