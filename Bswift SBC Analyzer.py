import streamlit as st
from PyPDF2 import PdfFileReader
from io import BytesIO
import time
import json
import boto3
import re

bedrock_client = boto3.client('bedrock-runtime')

def generate_message(bedrock_runtime, model_id, messages, max_tokens=4000, top_p=1, temp=0.5, system=''):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
            "system": system
        }
    )
    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())
    return response_body

def process_pdf(pdf_content):
    # Check if pdf_content is empty or contains only whitespace
    if not pdf_content.strip():
        return None, None, None

    messages = [{"role": 'user', "content": [{'type': 'text', 'text': pdf_content}]}]
    system = """
    You are a benefits admin helper for Bswift. You are given a Statement of Benefits and Coverages, and you need to help answer a set of questions

    <questions>
    What is the overall deductible? - Individual
    What is the overall deductible? - Family
    What is the out-of-pocket limit for this plan? - Individual
    What is the out-of-pocket limit for this plan? - Family
    If you visit a health care provider's office or clinic: Primary care visit to treat an injury or illness
    If you visit a health care provider's office or clinic: Specialist visit
    If you visit a health care provider's office or clinic: Preventive care/screening/ immunization
    If you have a test: Diagnostic test (x-ray, blood work)
    If you have a test: Diagnostic Lab Work
    If you have a test: Imaging (CT/PET scans, MRIs)
    If you have outpatient surgery: Physician/surgeon fees
    If you have a hospital stay: Physician/surgeon fees
    If you need mental health, behavioral health, or substance abuse services: Outpatient services
    If you need mental health, behavioral health, or substance abuse services: Inpatient services
    If you are pregnant: Childbirth/delivery professional services
    </questions>

    based on what is in the document. Your answers should either be a dollar amount or a % value based on what the beneficiary should pay. Some examples are the following:

    <example_answers>
    $2,250 individual
    $4,500 family
    $4,500 individual
    $9,000 family
    $10 copay per visit; deductible does not apply
    "$35 copay per visit;
    deductible does not apply"
    No charge
    20% coinsurance; deductible does not apply in outpatient setting
    20% coinsurance; deductible does not apply in outpatient setting
    "20% coinsurance; deductible does not
    apply"
    20% coinsurance after deductible
    20% coinsurance after deductible
    $10 copay per individual visit; $5 copay per group visit.
    20% coinsurance after deductible
    20% coinsurance after deductible
    </example_answers>

    Structure your response as a JSON code block with the following fields 
    question
    answer
    document_source (The text in the document you derived your answer from)
    page_number_of_source

    and put it between the <json></json> tags

    Then provide a markdown table with the Question and Answer from SBC in a customer presentation format (What you would show to a consumer of the medical benefits), and put it between the <markdown></markdown> tags.
    """

    response = generate_message(bedrock_client, model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                                messages=messages, max_tokens=4000, temp=0.5, top_p=0.9, system=system)

    response_text = response["content"][0]["text"]

    # Extract the JSON section
    json_match = re.search(r'<json>(.*?)</json>', response_text, re.DOTALL)
    json_data = None
    json_text = None
    if json_match:
        json_text = json_match.group(1)
        try:
            json_data = json.loads(json_text)
        except json.JSONDecodeError:
            pass

    # Extract the Markdown section
    markdown_match = re.search(r'<markdown>(.*?)</markdown>', response_text, re.DOTALL)
    markdown_data = markdown_match.group(1) if markdown_match else None

    return json_data, json_text, markdown_data

def extract_text_from_pdf(file):
    pdf_reader = PdfFileReader(BytesIO(file.read()))
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extractText()
    return text

def main():
    st.title("Bswift SBC Analyzer")

    # File upload button
    uploaded_file = st.file_uploader("Choose a SBC PDF file", type="pdf")

    if uploaded_file is not None:
        # Display success message
        st.success("Document uploaded successfully!")

        # Display a loading message while processing the PDF
        with st.spinner("Processing the SBC..."):
            # Extract text from the PDF
            pdf_content = extract_text_from_pdf(uploaded_file)

            # Process the PDF content
            json_data, json_text, markdown_data = process_pdf(pdf_content)

        if json_data is None and json_text is None and markdown_data is None:
            st.warning("The uploaded PDF does not contain any extractable text.")
        else:
            # Create two columns
            col1, col2 = st.columns(2)

            with col1:
                if json_data:
                    # Display the JSON data in the first column
                    st.subheader("Extracted Key Value Pairs")
                    st.json(json_data)
                elif json_text:
                    # Display the JSON text in the first column
                    st.subheader("Extracted Key Value Pairs")
                    st.text(json_text)
                else:
                    st.warning("No answers for the questions")

            with col2:
                if markdown_data:
                    # Display the Markdown data in the second column
                    st.subheader("Table Presentation")
                    st.markdown(markdown_data)
                else:
                    st.warning("No Markdown data found in the response.")

if __name__ == "__main__":
    main()