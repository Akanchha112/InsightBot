import os
from config import API_KEY

os.environ['GOOGLE_API_KEY'] = "AIzaSyC-8NsxbigOXxAMIKS2cV_2CkxXs79Y4iI "
# os.environ['GOOGLE_API_KEY'] = getpass.getpass('Gemini API Key:')

from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI


def get_user_url():
    """Prompts the user for a URL and returns it."""
    while True:
        url = input("Enter the URL of the article you want to summarize: ")
        if url.strip():  # Check if user entered a non-empty URL
            return url
        else:
            print("Please enter a valid URL.")


def summarize_article(url): 
    """Summarizes an article using the provided URL."""
    loader = WebBaseLoader(url)
    docs = loader.load()
  
    # To extract data from WebBaseLoader
    doc_prompt = PromptTemplate.from_template("{page_content}")

    # To query Gemini
    llm_prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:\n\n"""
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    # Create Stuff documents chain using LCEL
    stuff_chain = (
        # Extract data from the documents and add to the key `text`.
        {
            "text": lambda docs: "\n\n".join(
                format_document(doc, doc_prompt) for doc in docs
            )
        }
        | llm_prompt  # Prompt for Gemini
        | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85, google_api_key="AIzaSyC-8NsxbigOXxAMIKS2cV_2CkxXs79Y4iI ")  # This defines the llm variable
        | StrOutputParser()  # output parser
    )

    summary = stuff_chain.invoke(docs)
    print(summary)


# Get URL from user
url = get_user_url()

# Summarize the article
summarize_article(url)
