import re

with open('agents_graph.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Update PolicyState
code = code.replace("class PolicyState(TypedDict):\n    \"\"\"Shared state dictionary passed between all agents in the graph.\"\"\"\n    # Input\n    query: str", "class PolicyState(TypedDict):\n    \"\"\"Shared state dictionary passed between all agents in the graph.\"\"\"\n    api_key: str\n    # Input\n    query: str")

# 2. Update get_gemini signature
code = code.replace("def get_gemini(temperature: float = 0.3):", "def get_gemini(api_key: str = None, temperature: float = 0.3):")
code = code.replace("google_api_key=os.getenv(\"GOOGLE_API_KEY\"),", "google_api_key=api_key or os.getenv(\"GOOGLE_API_KEY\"),")

# 3. Update get_gemini calls to pass state.get("api_key")
code = re.sub(r"llm = get_gemini\((.*?)\)", r"llm = get_gemini(api_key=state.get('api_key'), \1)", code)
code = re.sub(r"llm = get_gemini\(\)", r"llm = get_gemini(api_key=state.get('api_key'))", code)

# 4. Fix run_query to accept api_key
code = code.replace("def run_query(app, query: str, chat_history: List = None):", "def run_query(app, query: str, api_key: str = None, chat_history: List = None):")
code = code.replace("\"query\": query,", "\"query\": query,\n        \"api_key\": api_key,")

with open('agents_graph.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Fixed agents_graph.py")
