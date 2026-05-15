from groq import Groq

# What: Create a client connected to Groq's servers
# Why: Groq hosts Llama 3 for free — same quality as GPT-4
client = Groq(api_key="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")

# What: Make an API call to Llama 3
# Why: This is the same pattern used in every production AI app
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Meta's Llama 3 — free and powerful
    messages=[
        {
            "role": "system",
            "content": "You are a helpful film industry analyst."
        },
        {
            "role": "user", 
            "content": "What are the top 3 factors that make a movie successful? Be concise."
        }
    ]
)

print(response.choices[0].message.content)