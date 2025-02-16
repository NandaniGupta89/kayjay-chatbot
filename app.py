import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from bs4 import BeautifulSoup
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_all_links(base_url):
    """Extracts all internal links from the given base URL."""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/") or base_url in href:  # Internal links only
            full_url = href if base_url in href else base_url + href
            links.add(full_url)

    return list(links)


def scrape_and_process_website():
    """Scrapes all pages from the Kay Jay Global website and processes them into FAISS."""
    base_url = "https://www.kayjayglobal.com/"
    urls = get_all_links(base_url)  # Get all page links

    all_text = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            all_text += page_text + "\n\n"
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    # Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Creates an AI response chain that uses only the Kay Jay Global website data."""
    prompt_template = """
    You are an AI assistant that provides well-structured, clear, and professional responses. 
    Base your answers only on the information available on the Kay Jay Global website: https://www.kayjayglobal.com/. 
    If the answer is not available on the website, state: 
    "The answer is not available on the Kay Jay Global website."

    ---
    Website Reference: https://www.kayjayglobal.com/
    Context:
    {context}

    Question:
    {question}

    ---
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    """Fetches the most relevant answer based on the scraped website data."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


def main():
    """Main Streamlit app interface."""
    st.set_page_config("Chat with Website")
    st.header("Chat with KayJay Global Website Using Gemini")
    user_question = st.text_input("Ask a Question about KayJay Global")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Scrape & Process Website"):
            with st.spinner("Scraping and Processing..."):
                scrape_and_process_website()
                st.success("Processing Complete")


if __name__ == "__main__":
    main()
