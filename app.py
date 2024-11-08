import streamlit as st
import torch 
from transformers import GPT2Tokenizer, LlamaForSequenceClassification
import fitz  # PyMuPDF library for PDF processing
import io
from torch.utils.data import Dataset
from sklearn.metrics import classification_report

# Load the tokenizer and model
model_path = "model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
model = LlamaForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    # Read the PDF file as a binary stream
    pdf_bytes = pdf_file.read()
    
    # Using BytesIO to convert the binary data into a file-like object
    pdf_stream = io.BytesIO(pdf_bytes)
    
    # Open the PDF using PyMuPDF from the file-like object
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    
    text = ""
    for page in doc:
        text += page.get_text("text")
    
    return text

# Function to preprocess and tokenize the input text
def preprocess_text(text1, text2):
    inputs = tokenizer(
        text1, text2,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return inputs

# Dataset class (similar to your existing one)
class PlagiarismDataset(Dataset):
    def __init__(self, text1, text2, tokenizer):
        self.text1 = text1
        self.text2 = text2
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        inputs = preprocess_text(self.text1[idx], self.text2[idx])
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

# Function to detect plagiarism using the model
def detect_plagiarism(text1, text2):
    dataset = PlagiarismDataset(text1, text2, tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.append(preds.item())

    return predictions[0]

# Streamlit UI
st.title("Plagiarism Detection using LLM")
st.write("Upload two PDFs for plagiarism detection.")

# Upload PDFs
pdf_file1 = st.file_uploader("Upload the first PDF", type="pdf")
pdf_file2 = st.file_uploader("Upload the second PDF", type="pdf")

if pdf_file1 and pdf_file2:
    # Extract text from PDFs
    text1 = extract_text_from_pdf(pdf_file1)
    text2 = extract_text_from_pdf(pdf_file2)

    # Display some text from the PDFs for context
    st.subheader("Text from the first document:")
    st.text(text1[:1000])  # Display the first 1000 characters of the document
    st.subheader("Text from the second document:")
    st.text(text2[:1000])

    # Detect plagiarism
    result = detect_plagiarism([text1], [text2])

    # Display the result
    if result == 1:
        st.success("Plagiarism detected!")
    else:
        st.success("No plagiarism detected.")
