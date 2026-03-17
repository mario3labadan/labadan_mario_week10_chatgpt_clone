### Task: Part 1A – Page Setup & API Connection
**Prompt:** Help me implement Part A of my Streamlit ChatGPT clone: set page config, load Hugging Face token from secrets, send a test message to the API, and handle errors gracefully. (copy and paste task 1a requirements)
**AI Suggestion:** The AI generated a structured Streamlit app that loads the Hugging Face token from st.secrets, sends a hardcoded message to the API, and displays the response. It also included comprehensive error handling for invalid tokens, rate limits, timeouts, and network issues.
**My Modifications & Reflections:** The code worked as expected after testing, and the API response displayed correctly in the app. I kept most of the structure but verified the token loading and ensured the error messages matched the assignment requirements for graceful failure

### Task: Part 1B – Multi-Turn Conversation UI
**Prompt:** Help me extend Part A into a multi-turn Streamlit chat app using st.chat_message and st.chat_input, store conversation history in st.session_state, send the full message history to the Hugging Face API, and handle errors without crashing. (copy and paste task 1b requirements)
**AI Suggestion:** The AI rewrote the app to use Streamlit’s native chat components, added a session-state messages list, and sent the full conversation history with each API call so the model could keep context. It also placed the chat history in a container above the fixed input bar and reused graceful API error handling from Part A.
**My Modifications & Reflections:** The code mostly worked and successfully preserved conversation context across multiple messages. I tested the session-state behavior and kept the structure. I told it my name, then asked what my name was to make sure it remebered. 

### Task: Part 1C – Chat Management
**Prompt:** Help me add sidebar-based chat management to my Streamlit app with a New Chat button, a scrollable chat list, active chat highlighting, chat switching, and delete buttons for each chat. (copy and paste task 1c requirements)
**AI Suggestion:** The AI added a sidebar chat system using st.sidebar, where each chat is stored as its own record with an ID, title, timestamp, and message history. It also implemented chat creation, switching, deletion, and active chat highlighting, while preserving each conversation independently.
**My Modifications & Reflections:** The code worked well after testing, and I confirmed that chats could be created, switched, and deleted without affecting the others. I kept most of the logic but checked that the active chat updated correctly after deletion and that the sidebar behavior matched the assignment requirements.

### Task: Part 1D – Chat Persistence
**Prompt:** Help me implement chat persistence by saving each chat as a JSON file in a chats/ directory, loading them on startup, allowing continued conversations, and deleting files when chats are removed. (copy and paste task 1d requirements)
**AI Suggestion:** The AI added file-based persistence using JSON, where each chat is saved and loaded from the chats/ directory with its ID, title, timestamp, and message history. It also integrated saving after each message and deletion of the corresponding file when a chat is removed.
**My Modifications & Reflections:** The code worked correctly after testing, and chats persisted even after restarting the app. I verified that loading, continuing conversations, and deleting chats all behaved as expected, ensuring it met the assignment’s persistence requirements.

### Task: Task 2 – Response Streaming
**Prompt:** Help me update my Streamlit chat app so the model response streams incrementally using the Hugging Face API with stream=True, shows chunks live in the UI, and saves the full streamed reply to chat history after completion.(copy and paste task 2 requirements)
**AI Suggestion:** The AI replaced the normal response function with a streaming generator that reads server-sent event chunks from the Hugging Face API and yields text progressively. It used st.write_stream() to display the reply live in the chat interface and added a short delay so the streaming effect would be visible.
**My Modifications & Reflections:** The code worked after testing, and the assistant response appeared incrementally instead of all at once. I verified that the full final response was still saved correctly to the chat history and persisted like the earlier non-streamed version.

### Task: Task 3 – User Memory
**Prompt:** Help me add persistent user memory to my Streamlit chat app by extracting traits or preferences from user messages, saving them in memory.json, showing them in the sidebar, allowing memory reset, and injecting that memory into future prompts for personalization.
**AI Suggestion:** The AI added a second API call that extracts user traits from each message as a JSON object, then merges and saves that data in memory.json. It also displayed the saved memory in a sidebar expander, added a clear-memory button, and included the stored memory in a system prompt for future conversations.
**My Modifications & Reflections:** The feature worked after testing, and I confirmed that saved traits appeared in the sidebar and influenced later responses. I kept the overall structure but verified that memory persisted across app restarts and that clearing memory reset both the file and the sidebar display.
