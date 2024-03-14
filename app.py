import docx
import os
from flask import Flask, request, render_template
from openai import OpenAI
import PyPDF2
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# client = OpenAI(api_key="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")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return 'No file uploaded', 400

    resume_file = request.files['resume']

    if resume_file.filename == '':
        return 'No file selected', 400

    # Check file extension and process accordingly
    if resume_file.filename.endswith('.pdf'):
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_file.filename.endswith('.doc') or resume_file.filename.endswith('.docx'):
        resume_text = extract_text_from_docx(resume_file)
    else:
        return 'Unsupported file format', 400

    feedback = generate_feedback(resume_text)

    return render_template('feedback.html', feedback=feedback)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def generate_feedback(text):
    prompt = "The following is the resume content of a candidate. Please provide feedback on the candidate for the position of Software Engineer. " + "\n"+ text
    # response = client.completions.create(model="gpt-3.5-turbo-instruct", prompt=full_text, temperature=0.4, max_tokens=150)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.8, max_length=1000)
    feedback = tokenizer.batch_decode(gen_tokens)[0]
    return feedback

if __name__ == '__main__':
    app.run(debug=True)
