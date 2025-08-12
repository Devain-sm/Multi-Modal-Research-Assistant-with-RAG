import os
import faiss
import numpy as np
from PIL import Image
import pytesseract
import pypdf
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List, Dict, Any
import uvicorn

# --- 1. Configuration & Model Loading ---

# Load a pre-trained multi-modal embedding model from Hugging Face
print("Loading embedding model...")
# Using a model that's good for both text and image-like semantics
model = SentenceTransformer('clip-ViT-B-32')
embedding_dim = model.get_sentence_embedding_dimension()

# Initialize a FAISS index for vector storage and retrieval
# IndexFlatL2 is a simple L2 (Euclidean) distance search
index = faiss.index_factory(embedding_dim,"Flat")

# In-memory storage to map FAISS indices back to content
# In a real app, you'd use a persistent database like PostgreSQL or a document store
document_store: Dict[int, Dict[str, Any]] = {}
doc_counter = 0

# Create a directory for uploaded files
os.makedirs("uploads", exist_ok=True)


# --- 2. Document Ingestion & Processing ---

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    print(f"Extracting text from PDF: {file_path}")
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        text = "".join(page.extract_text() for page in reader.pages)
    return text

def extract_text_from_image(file_path: str) -> str:
    """Extracts text from an image file using OCR (Tesseract)."""
    print(f"Extracting text from Image: {file_path}")
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
        return text
    except Exception as e:
        print(f"Error processing image with OCR: {e}")
        return ""

def add_to_vector_db(chunks: List[str], source_file: str, content_type: str):
    """Encodes text chunks and adds them to the FAISS index."""
    global doc_counter
    print(f"Generating embeddings for {len(chunks)} chunks from {source_file}...")
    
    # Generate embeddings for the chunks
    embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)
    
    # Add embeddings to the FAISS index
    index.add(np.array(embeddings).astype('float32'))
    
    # Store the original content and metadata
    for chunk in chunks:
        document_store[doc_counter] = {
            "content": chunk,
            "source": source_file,
            "type": content_type
        }
        doc_counter += 1
    
    print(f"Successfully added {len(chunks)} chunks to DB. Total docs: {index.ntotal}")

# --- 3. Retrieval-Augmented Generation (RAG) Logic ---

def query_rag(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs the RAG process: query, retrieve, augment."""
    if index.ntotal == 0:
        return []

    print(f"1. Encoding query: '{query}'")
    query_embedding = model.encode([query])
    
    print(f"2. Retrieving top {top_k} relevant documents from FAISS index...")
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    # Retrieve the actual content from our document store
    results = [document_store[i] for i in indices[0] if i in document_store]
    
    print(f"3. Found {len(results)} relevant documents.")
    return results

def generate_answer(query: str, context: List[Dict[str, Any]]) -> str:
    """
    Simulates the 'Generation' part of RAG.
    In a real system, this would feed the context and query into an LLM (e.g., GPT, Llama).
    
    **Advanced Twist Note:** A true cross-modal attention model (built in PyTorch) would
    operate here. It wouldn't just concatenate text but would intelligently weigh image
    features against text features when generating the final answer.
    """
    if not context:
        return "I could not find any relevant information in the uploaded documents to answer your question."

    # Simple augmentation: just prepend context to the query for the LLM
    prompt = "Based on the following context, please answer the question.\n\n"
    prompt += "--- Context ---\n"
    for item in context:
        prompt += f"Source: {item['source']} ({item['type']})\nContent: {item['content']}\n\n"
    prompt += "--- Question ---\n"
    prompt += query + "\n\n--- Answer ---\n"
    
    # Here, you would call your LLM API. We will simulate it.
    print("\n--- Simulated LLM Prompt ---")
    print(prompt)
    print("---------------------------\n")

    return f"This is a simulated answer. An LLM would use the above prompt to generate a detailed response. The most relevant piece of context found was from '{context[0]['source']}'."


# --- 4. FastAPI Application ---

app = FastAPI(
    title="Multi-Modal RAG Research Assistant",
    description="Upload PDFs and images, then ask questions about their content.",
)

@app.post("/upload/", summary="Upload a document (PDF or Image)")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests a document, processes its content, and adds it to the vector DB.
    """
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    content_type = file.content_type
    text_content = ""
    
    if content_type == "application/pdf":
        text_content = extract_text_from_pdf(file_path)
    elif content_type in ["image/png", "image/jpeg", "image/jpg"]:
        # For images, we get OCR text AND we can also embed the image itself
        text_content = extract_text_from_image(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or an image (PNG, JPG).")

    if not text_content:
        raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

    # Simple chunking strategy (split by paragraph)
    chunks = [p.strip() for p in text_content.split("\n\n") if p.strip()]
    
    add_to_vector_db(chunks, source_file=file.filename, content_type=content_type)
    
    return {"status": "success", "filename": file.filename, "chunks_added": len(chunks)}


@app.post("/query/", summary="Ask a question about the uploaded documents")
async def query_assistant(query: str = Form(...)):
    """
    Takes a user query, finds relevant context using RAG, and generates an answer.
    """
    print(f"\nReceived query: '{query}'")
    retrieved_context = query_rag(query, top_k=3)
    
    answer = generate_answer(query, retrieved_context)
    
    return {"query": query, "answer": answer, "retrieved_context": retrieved_context}


@app.get("/", summary="Root endpoint")
async def root():
    return {"message": "Welcome to the Multi-Modal RAG Assistant API. Go to /docs to see the endpoints."}


# --- 5. Run the Application ---

if __name__ == "__main__":
    print("Starting FastAPI server...")
    # To run: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)