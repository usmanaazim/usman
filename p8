!pip -q install langchain langchain-community cohere

import os
from langchain_community.llms import Cohere
from langchain_core.prompts import PromptTemplate

os.environ["COHERE_API_KEY"] = "YOUR_API_KEY_HERE"

llm = Cohere(model="command-light", temperature=0.5)

prompt = PromptTemplate.from_template("""
Provide structured output:
Summary:
Key Points:
Conclusion:
{text}
""")

text = """Artificial Intelligence is transforming healthcare by improving diagnosis and prediction."""

try:
    response = llm.invoke(prompt.format(text=text))
    print("✅ MODEL OUTPUT:\n")
    print(response)
except Exception:
    print("⚠️ Cohere failed, showing fallback output\n")
    print("""
Summary:
Artificial Intelligence is improving healthcare systems.

Key Points:
- Used in diagnosis
- Helps prediction
- Enhances efficiency

Conclusion:
AI is transforming healthcare significantly.
""")
