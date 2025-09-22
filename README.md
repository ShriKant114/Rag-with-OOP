````markdown
# RAG + Gemini Chatbot Project

This project uses **RAG (Retrieval-Augmented Generation)** to process PDF documents and generate answers to user questions using **Google Gemini LLM**.  

It stores embeddings in **Pinecone** and uses **Sentence-Transformers (MPNet)** for vector embeddings.

---

## 📦 Requirements

- Python 3.10+  
- Pip packages (install all together):

```bash
pip install -r requirements.txt
````

`requirements.txt` content:

```
google-generativeai
pinecone-client>=2.2.0
sentence-transformers
pdfplumber
python-dotenv
```

---

## 🔑 Step 1: Gemini API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Navigate to **AI & ML → Generative AI → Credentials → Create API Key**.
3. Copy the API key.
4. Directly configure it in the Python code (no `.env` needed):

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

## ☁️ Step 2: Pinecone Setup

1. Go to [Pinecone.io](https://www.pinecone.io/) and create a free account.
2. Generate your **API key**.
3. Use your existing index or create a new one:

```python
import pinecone

pinecone_api_key = "YOUR_PINECONE_API_KEY"
pc = pinecone.Pinecone(api_key=pinecone_api_key)
index = pc.Index("rag768")  # Use your index name
```

**Index configuration:**

* Dimension: 768 (MPNet embedding)
* Metric: cosine
* Cloud: AWS
* Region: us-east-1

---

## 📄 Step 3: Project Structure

```
project/
│
├── opp.pdf             # Your PDF document
├── main.py             # Main RAG + Gemini script
├── requirements.txt    # Pip packages
└── README.md           # This file
```

---

## 📝 Step 4: Running the Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main script:

```bash
python main.py
```

3. Start chatting:

```text
✅ RAG + Gemini Chat ready. Type 'exit' to quit.

You: Benefits of OOP
Gemini: OOP provides modularity, reusability, and maintainability...
```

4. Type `exit` to quit.

---

## ⚙️ How it works

1. **PDF loading & chunking**: 500 words per chunk.
2. **Embeddings**: Each chunk is converted to a 768-dim vector using `sentence-transformers/all-mpnet-base-v2`.
3. **Pinecone**: Stores embeddings for retrieval.
4. **RAG**: Top 3 relevant chunks are retrieved for a user question.
5. **Gemini LLM**: Prompt is sent to `models/gemma-3-4b-it` (Gamma model) to generate answer.

---

## ✅ Notes

* No `.env` file is required; API keys can be configured directly in code.
* Make sure your **Gemini API key** and **Pinecone API key** are valid.
* Ensure your Pinecone index dimension matches embedding size (768).

---

## 📚 References

* [Google Gemini API Docs](https://developers.generativeai.google/)
* [Pinecone Documentation](https://docs.pinecone.io/)
* [Sentence-Transformers](https://www.sbert.net/)

```
