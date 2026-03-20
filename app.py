import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = [
    "Construction delays happen due to labor shortages.",
    "Weather conditions affect project timelines.",
    "Material supply issues cause delays."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

st.title("Mini RAG Chatbot")

query = st.text_input("Ask your question")

if query:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), 2)
    retrieved = [docs[i] for i in indices[0]]

    st.write("### Retrieved Context")
    for r in retrieved:
        st.write("-", r)

    context = " ".join(retrieved)

    if context:
        answer = "Answer: " + context
    else:
        answer = "Not available"

    st.write("### Final Answer")
    st.write(answer)
