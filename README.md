````markdown
# RAG + Gemini Chatbot Project

This project uses **RAG (Retrieval-Augmented Generation)** to process PDF documents and generate answers to user questions using **Google Gemini LLM**.  

It stores embeddings in **Pinecone** and uses **Sentence-Transformers (MPNet)** for vector embeddings.

---

## 📦 Requirements

- Python 3.10+  
- Pip packages (install all at once):

```bash
pip install -r requirements.txt
````

---

## 🔑 Step 1: Gemini API Key

1. Go to [Google AI Studio Cloud Console](https://aistudio.google.com/app/apikey).
2. Navigate to **AI & ML → Generative AI → Credentials → Create API Key**.
3. Copy the API key.
4. Directly configure it in the Python code (no `.env` needed):

```python
Replace this with your api key 

genai.configure(api_key="ADD YOUR API KEY PLEASE REFER HOW TO SET UP IN THE README.MD FILE"
```

---

## ☁️ Step 2: Pinecone Setup

1. Go to [Pinecone.io](https://www.pinecone.io/) and create a free account.
2. Generate your **API key**.
3. Use your existing index or create a new one:

```python

Replace this with your API KEY

pinecone_api_key = "ADD YOUR API KEY PLEASE REFER HOW TO SET UP IN THE README.MD FILE"

```

**Index configuration:**

* Dimension: 768 (MPNet embedding)
* Metric: cosine
* Cloud: AWS
* Region: us-east-1

> ⚠️ Make sure the dimension of the Pinecone index matches the embedding size (768 for MPNet).

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

4. Type `exit` to quit the chat.

---

## ⚙️ How it works

1. **PDF loading & chunking**: 500 words per chunk.
2. **Embeddings**: Each chunk is converted to a 768-dim vector using `sentence-transformers/all-mpnet-base-v2`.
3. **Pinecone**: Stores embeddings for retrieval.
4. **RAG**: Top 3 relevant chunks are retrieved for a user question.
5. **Gemini LLM**: Prompt is sent to `models/gemma-3-4b-it` (Gamma model) to generate the answer.


---

## 📚 References

* [Google Gemini API Docs](https://developers.generativeai.google/)
* [Pinecone Documentation](https://docs.pinecone.io/)
* [Sentence-Transformers](https://www.sbert.net/)

```
