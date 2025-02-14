import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Ensure API key is set
if not gemini_api_key:
    st.error("❌ Missing Gemini API Key! Set GEMINI_API_KEY in your .env file.")
    st.stop()

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            # Add filename as metadata.  Important!
            yield text, {"source": pdf.name} #Use a generator.  Yield returns text and metadata.

        except Exception as e:
            st.warning(f"⚠️ Could not read PDF: {pdf.name}. Skipping. Error: {e}")
            yield "", {"source": pdf.name, "error": str(e)} #Yield a blank text and metadata for error too.

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create vector store with embeddings
def get_vectorstore(pdf_docs):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=gemini_api_key
        )

        all_texts = []
        all_metadatas = []

        for text, metadata in get_pdf_text(pdf_docs):
           if text: #Make sure that text is available
               all_texts.append(text)
               all_metadatas.append(metadata)

        vectorstore = FAISS.from_texts(all_texts, embedding=embeddings, metadatas=all_metadatas)
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error initializing vector store: {e}")
        return None

# Function to create conversation chain (without memory this time)
def get_conversation_chain(vectorstore):
    try:
        # Use a model known to be available
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",  # <---  USE THIS OR A VALID MODEL
            google_api_key=gemini_api_key
        )

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
        return conversation_chain
    except Exception as e:
        st.error(f"❌ Error initializing conversation chain: {e}")
        return None

# Function to handle user input (modified for manual chat history)
def handle_userinput(user_question):
    try:
        if st.session_state.conversation:
            # Get the response from the chain
            response = st.session_state.conversation({"question": user_question, "chat_history": st.session_state.chat_history}) #Pass chat history

            # Add the user's question to chat history
            st.session_state.chat_history.append({"type": "human", "content": user_question})

            # Add the bot's answer to chat history, handling potential None answers from the model
            if response and "answer" in response and response["answer"]:
                st.session_state.chat_history.append({"type": "ai", "content": response["answer"]})

            # Display chat history, handling potential None values
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    # Display messages directly using templates (removes the extra $MSG$ replacements)
                    if message.get("type") == "human":
                        st.write(user_template.replace("{{MSG}}", message.get("content", "")), unsafe_allow_html=True)
                    elif message.get("type") == "ai":
                        st.write(bot_template.replace("{{MSG}}", message.get("content", "")), unsafe_allow_html=True)

        else:
            st.error("❌ Please process your PDFs first.")
    except Exception as e:
        st.error(f"❌ Error handling user input: {e}")

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    # Session State Initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize with an empty list

    st.subheader("Chat with Multiple PDFs 📚")


    # User input with a "Send" button
    with st.form(key='my_form'):  # Use a form
        user_question = st.text_input("Ask a question about your documents:")
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_question and st.session_state.conversation:
            #handle_userinput(modified_question) # Removed user name prepending
             handle_userinput(user_question) #Pass the original question
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True, type=["pdf"])

        if st.button("Process"):
            if not pdf_docs:
                st.error("❌ Please upload at least one PDF file.")
                return

            with st.spinner("🔄 Processing PDFs..."):
                # Create vector store, passes pdf_docs
                vectorstore = get_vectorstore(pdf_docs)

                if vectorstore:
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)  # Initialize
                    st.success("✅ PDFs processed successfully!")
                    # Clear chat history when PDFs are re-processed
                    st.session_state.chat_history = []

if __name__ == "__main__":
    main()






# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS

# # Load environment variables
# load_dotenv()

# # Set API key explicitly
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# if not gemini_api_key:
#     st.error("Missing Gemini API Key! Set GEMINI_API_KEY in .env")
#     st.stop()  # Stop execution if API key is missing

# # Ensure the API key is correctly set for langchain_google_genai
# os.environ["GOOGLE_API_KEY"] = gemini_api_key  # Explicitly set in environment

# # Function to extract text from PDFs
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""  # Handle None values
#     return text

# # Function to split text into chunks
# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     return text_splitter.split_text(text)

# # Function to create vector store with embeddings
# def get_vectorstore(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # ✅ Correct model name
#     return FAISS.from_texts(text_chunks, embedding=embeddings)

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")

#     st.title("Chat with Multiple PDFs 📚")
#     query = st.text_input("Ask a question about your documents")

#     with st.sidebar:
#         st.subheader("Upload Your PDFs")
#         pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True, type=["pdf"])

#         if st.button("Process"):
#             if not pdf_docs:
#                 st.error("Please upload at least one PDF file.")
#                 return
            
#             with st.spinner("Processing..."):
#                 # Extract text from PDFs
#                 raw_text = get_pdf_text(pdf_docs)

#                 # Split text into chunks
#                 text_chunks = get_text_chunks(raw_text)

#                 # Create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # Store vectorstore in session state (optional)
#                 st.session_state["vectorstore"] = vectorstore  

#                 # Display status
#                 st.success(f"Processed {len(text_chunks)} text chunks. Vector store created successfully!")

# if __name__ == "__main__":
#     main()
