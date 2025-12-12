import os
import sys

print("1. Starting script...")

try:
    from dotenv import load_dotenv
    print("2. Imported dotenv")
    
    from langchain_openai import ChatOpenAI
    print("3. Imported OpenAI")
    
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    print("4. Imported LangChain modules")

except ImportError as e:
    print(f"CRITICAL ERROR: Missing library. {e}")
    sys.exit()

# 1. Load Secrets
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: .env file not found or empty!")
    sys.exit()

print("5. API Key loaded")

# 2. Setup the "Brain"
print("6. Initializing Model...")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 3. Create the Memory Store
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 4. System Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Customer Support Agent for 'TechGear'. "
            "You are helpful but FIRM. "
            "POLICY: You can only process refunds for orders made within 30 days. "
            "If an order is older than 30 days, you must politely decline."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# 5. Chain setup
print("7. Setting up Chain...")
runnable = prompt | model

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 6. Chat Loop
print("\n--- TechGear Support Bot (System Ready) ---")
print("Bot: Hello! How can I help you? (Type 'quit' to exit)")

session_id = "user_123"

while True:
    # We force the print to show up immediately
    try:
        user_input = input("\nYou: ")
    except KeyboardInterrupt:
        break
        
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    print("Bot is thinking...")
    try:
        response = with_message_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Bot: {response.content}")
    except Exception as e:
        print(f"API ERROR: {e}")