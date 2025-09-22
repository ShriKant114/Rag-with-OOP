import os
import pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pdfplumber

# -------------------------------
# 1️⃣ Configure Gemini
# -------------------------------
gemini_api_key = "ADD YOUR API KEY PLEASE REFER HOW TO SET UP IN THE README.MD FILE"
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel('models/gemma-3-4b-it')

# -------------------------------
# 2️⃣ Load embedding model (MPNet, 768-dim)
# -------------------------------
embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_embedding(text):
    return embed_model.encode(text).tolist()  # 768-dim vector

# -------------------------------
# 3️⃣ Pinecone setup (768-dim index)
# -------------------------------
pinecone_api_key = "ADD YOUR API KEY PLEASE REFER HOW TO SET UP IN THE README.MD FILE"
pinecone_env = "us-east-1"

pc = pinecone.Pinecone(api_key=pinecone_api_key)

# ✅ Naya index check & create (768-dim)
index_name = "rag768"
if index_name not in pc.list_indexes().names():
    from pinecone import ServerlessSpec
    pc.create_index(
        name=index_name,
        dimension=768,  # MPNet embedding dimension
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=pinecone_env)
    )

index = pc.Index(index_name)

# -------------------------------
# 4️⃣ PDF load & chunking
# -------------------------------
pdf_path = "opp.pdf"
chunks = []

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            words = text.split()
            for i in range(0, len(words), 500):
                chunk_text = " ".join(words[i:i+500])
                chunks.append(chunk_text)

# -------------------------------
# 5️⃣ Upsert chunks to Pinecone
# -------------------------------
for i, chunk in enumerate(chunks):
    vector = get_embedding(chunk)
    index.upsert([(f"id-{i}", vector, {"text": chunk})])

print(f"✅ {len(chunks)} chunks uploaded to Pinecone.")

# -------------------------------
# 6️⃣ RAG + Gemini interactive chat
# -------------------------------
def retrieve_chunks(question, top_k=3):
    q_vector = get_embedding(question)
    res = index.query(vector=q_vector, top_k=top_k, include_metadata=True)
    return [item['metadata']['text'] for item in res['matches']]

print("✅ RAG + Gemini Chat ready. Type 'exit' to quit.\n")

# Gemini chat start
chat = gemini_model.start_chat(history=[])

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # 1️⃣ Retrieve relevant chunks from Pinecone
    top_chunks = retrieve_chunks(user_input, top_k=3)

    # 2️⃣ Prepare prompt
    if len(top_chunks) == 0:
        prompt = "Sorry this question ans is not avilabal in our dataset"
    else:
        prompt = "Answer the question based on context:\n"
        for c in top_chunks:
            prompt += c + "\n"
        prompt += "\nQuestion: " + user_input

    # 3️⃣ Send prompt to Gemini chat
    try:
        response = chat.send_message(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Error: {e}"

    print("Gemini:", answer)

