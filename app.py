import streamlit as st
import requests

# ── What: Page configuration
# ── Why: Sets the browser tab title and page layout
st.set_page_config(
    page_title="Movie RAG Chatbot",
    page_icon="🎬",
    layout="centered"
)

# ── What: App title and description
# ── Why: st.title, st.markdown are Streamlit's way of adding text to the UI
st.title("🎬 Movie RAG Chatbot")
st.markdown("Ask me anything about 500 movies from the TMDB dataset!")

# ── What: Session state for chat history
# ── Why: Streamlit reruns the entire script on every interaction
#         Session state persists data between reruns — like memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── What: Display chat history
# ── Why: Show all previous messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── What: Chat input box
# ── Why: st.chat_input creates the message box at the bottom
#         Returns the user's message when they press Enter
if prompt := st.chat_input("Ask about movies..."):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # ── What: Call the FastAPI backend
    # ── Why: Streamlit is just the UI — the actual RAG logic
    #         lives in the FastAPI server running on port 8000
    with st.chat_message("assistant"):
        with st.spinner("Searching movies and thinking..."):
            try:
                # Send POST request to FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"text": prompt}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display sources in an expandable section
                    with st.expander("📚 Sources used"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**{i}.** {source}")
                else:
                    answer = "Sorry, something went wrong. Is the API server running?"
                    st.markdown(answer)
                    
            except Exception as e:
                answer = f"Cannot connect to API server. Make sure api.py is running! Error: {str(e)}"
                st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── What: Sidebar with info
# ── Why: Good UX practice — show users what they can ask
with st.sidebar:
    st.header("About")
    st.markdown("This chatbot uses **RAG** (Retrieval Augmented Generation) to answer questions about movies.")
    st.markdown("---")
    st.header("Example questions")
    st.markdown("- Tell me about Avatar")
    st.markdown("- What movies involve superheroes?")
    st.markdown("- Which movies are about space?")
    st.markdown("- What is Inception about?")
    st.markdown("---")
    st.header("Tech stack")
    st.markdown("- **LLM:** Llama 3.3 via Groq")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector store:** FAISS")
    st.markdown("- **Backend:** FastAPI")
    st.markdown("- **Frontend:** Streamlit")