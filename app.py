import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load .env for local development. On Streamlit Community Cloud, use Secrets.
load_dotenv(override=False)

st.set_page_config(
    page_title="Drug RAG Assistant",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Drug Label Assistant")
st.caption("Ask questions about the metformin extended-release drug label.")
st.warning(
    "For educational information only. This app is not a substitute for "
    "professional medical advice. Confirm medication decisions with a qualified "
    "healthcare professional."
)

# ---- Configuration ----
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "metformin_labels")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openrouter/free")
N_RESULTS = int(os.getenv("RAG_N_RESULTS", "3"))

# This is an initial guardrail, not a universal ChromaDB cutoff.
# Tune it using known relevant and irrelevant questions.
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    st.error(
        "OpenRouter API key not found. For local use, add "
        "`OPENROUTER_API_KEY=your_key` to a `.env` file. "
        "For Streamlit Community Cloud, add it under App settings → Secrets."
    )
    st.stop()

@st.cache_resource
def load_resources(chroma_path: str, collection_name: str):
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    db_client = chromadb.PersistentClient(path=chroma_path)
    db_collection = db_client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )
    return db_client, db_collection

try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    db_client, collection = load_resources(CHROMA_PATH, COLLECTION_NAME)
except Exception as exc:
    st.error(
        "Could not initialize the app. Check that the `chroma_db` folder is "
        "available, the collection exists, and the API key is valid."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()


def retrieve_chunks(question: str, n_results: int = N_RESULTS):
    """Route obvious question types to a relevant label section, otherwise search generally."""
    q = question.lower()

    routes = [
        (("contraindication", "contraindications"), "4 CONTRAINDICATIONS"),
        (("meal", "food", "dosage", "dose", "administer", "administration"),
         "2.1 Adult Dosage and Administration"),
        (("adverse reaction", "adverse reactions", "side effect", "side effects"),
         "6.1 Clinical Studies Experience"),
        (("lactic acidosis",), "WARNING: LACTIC ACIDOSIS"),
    ]

    target_section = None
    for keywords, section in routes:
        if any(keyword in q for keyword in keywords):
            target_section = section
            break

    query_args = {
        "query_texts": [question],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if target_section:
        query_args["where"] = {"section": target_section}

    results = collection.query(**query_args)

    # Fall back to general semantic search if a routed section has no records.
    if target_section and not results["documents"][0]:
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    return results, target_section


def build_context(results):
    docs = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []
    distances = results["distances"][0] or []
    sources = []
    context_parts = []

    for index, (doc, metadata, distance) in enumerate(
        zip(docs, metadatas, distances), start=1
    ):
        section = metadata.get("section", "Unknown section")
        drug = metadata.get("drug", "Unknown drug")
        context_parts.append(
            f"SOURCE {index}\nDrug: {drug}\nSection: {section}\n"
            f"Distance: {distance:.4f}\nExcerpt:\n{doc}"
        )
        sources.append(
            {
                "section": section,
                "drug": drug,
                "distance": float(distance),
                "text": doc,
            }
        )

    return "\n\n".join(context_parts), sources


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
Answer the user's question using ONLY the official drug-label excerpts below.

QUESTION:
{question}

SOURCE EXCERPTS:
{context}

Rules:
- Answer the exact question asked and include only relevant information.
- Do not invent medical facts, treatment recommendations, section numbers,
  citations, or facts from outside the supplied excerpts.
- Preserve medical terminology, numbers, percentages, and qualifiers.
- For adverse-reaction questions, preserve frequency groups exactly as stated.
  Include all reactions and percentages that appear in the excerpts.
- Keep laboratory findings separate from adverse reactions when the source does.
- Do not infer that an unfamiliar or fictional condition is related to a
  condition mentioned in the excerpts.
- If the excerpts do not answer the question, say:
  "The retrieved excerpts do not contain enough information to answer this question."
- Mention the source section when useful.
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful drug-label information assistant. "
                    "Use only the supplied excerpts and be explicit about gaps."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or (
        "The model returned an empty answer."
    )


# ---- Sidebar ----
with st.sidebar:
    st.header("About this app")
    st.write("**Knowledge source:** Metformin hydrochloride extended-release label")
    st.write(f"**LLM:** `{MODEL_NAME}`")
    st.write("**Retrieval:** ChromaDB + all-MiniLM-L6-v2")
    st.caption(
        "Prototype note: answers depend on the available label excerpts and "
        "should be verified against the official label."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Retrieved source sections"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**{i}. {source['section']}** — "
                        f"distance `{source['distance']:.4f}`"
                    )
                    with st.expander(f"View excerpt {i}"):
                        st.write(source["text"])

question = st.chat_input("Ask a question about the metformin label...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the drug label and generating an answer..."):
            try:
                results, routed_section = retrieve_chunks(question)
                context, sources = build_context(results)

                if not sources:
                    answer = (
                        "No relevant excerpts were retrieved from the current "
                        "drug-label database, so I can't answer this from the "
                        "available source."
                    )
                elif min(source["distance"] for source in sources) > MAX_DISTANCE:
                    answer = (
                        "I couldn't find sufficiently relevant excerpts in the "
                        "current drug-label database to answer this reliably. "
                        "Try rephrasing the question or checking the official label."
                    )
                else:
                    answer = generate_answer(question, context)

                st.markdown(answer)

                if sources:
                    with st.expander("Retrieved source sections"):
                        for i, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**{i}. {source['section']}** — "
                                f"distance `{source['distance']:.4f}`"
                            )
                            with st.expander(f"View excerpt {i}"):
                                st.write(source["text"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                st.error(
                    "Something went wrong while retrieving or generating the answer."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))
