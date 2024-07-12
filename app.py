import os
import logging
from flask import Flask, render_template, request, jsonify
from langchain_core.prompts import PromptTemplate  # Updated import
from langchain_community.document_loaders import WebBaseLoader  # Updated import
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route("/")
def start():
    return render_template('index.html')

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    url = data.get("url")
    logging.debug(f"Received URL: {url}")
    if not url:
        logging.error("URL is missing")
        return jsonify({"error": "URL is required"}), 400

    try:
        summary = summarize_article(url)
        logging.debug(f"Generated summary: {summary}")
        return jsonify({"summary": summary})
    except Exception as e:
        logging.exception("An error occurred during summarization")
        return jsonify({"error": str(e)}), 500

def summarize_article(url):
    # print("Heeloo o am inside summarize")
    loader = WebBaseLoader(url)
    docs = loader.load()
    logging.debug(f"Loaded documents: {docs}")

    doc_prompt = PromptTemplate.from_template("{page_content}")

    llm_prompt_template = """Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:\n\n"""
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    stuff_chain = (
        {
            "text": lambda docs: "\n\n".join(
                format_document(doc, doc_prompt) for doc in docs
            )
        }
        | llm_prompt
        | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85, google_api_key="AIzaSyC-8NsxbigOXxAMIKS2cV_2CkxXs79Y4iI")
        | StrOutputParser()
    )

    summary = stuff_chain.invoke(docs)
    logging.debug(f"Summary from LLM: {summary}")
    return summary

if __name__ == "__main__":
    app.run(debug=True, port=5001)
