 RAGnify – Retrieval Augmented Generation Project

 Project Overview
RAGnify is a Retrieval Augmented Generation (RAG) application that allows users to ask questions based on uploaded PDF documents and receive relevant answers using AI.

 Technologies Used
- Python
- Flask
- LangChain
- FAISS Vector Database
- Hugging Face Embeddings
- Google Gemini API
- PyPDF

 Key Features
- PDF document loading and text extraction
- Text splitting into chunks
- Text embedding generation
- Vector storage using FAISS
- Similarity search for relevant information
- Question answering using an LLM
- Context-based responses

Project Workflow
1. Upload or load a PDF document.
2. Extract text from the PDF.
3. Split text into smaller chunks.
4. Generate embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on the question.
7. Generate an answer using Gemini.

 How to Run
1. Clone the repository.
2. Install dependencies using pip install -r requirements.txt.
3. Configure your Gemini API key in the environment file.
4. Run the Flask application using python app.py.
 Author
Python and Generative AI Project
