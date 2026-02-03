from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="moonshotai/Kimi-K2.5",
    provider="hf-inference"
)

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2.5",
    task="text-generation",
    client=client
)

model = ChatHuggingFace(llm=llm)

print(model.invoke("Explain about ai?").content)
