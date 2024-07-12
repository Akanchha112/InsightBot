import os
from flask import Flask, render_template, request, jsonify
from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

@app.route("/")
def start():
    return "The MBSA Server is Running"

@app.route("/mbsa")
def mbsa():
    return render_template('index.html')

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        summary = summarize_article(url)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def summarize_article(url): 
    loader = WebBaseLoader(url)
    docs = loader.load()
    
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
        | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85, google_api_key=os.getenv("GOOGLE_API_KEY"))
        | StrOutputParser()
    )
    
    summary = stuff_chain.invoke(docs)
    return summary

if __name__ == "__main__":
    app.run(debug=True)
