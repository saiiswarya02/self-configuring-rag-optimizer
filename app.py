import streamlit as st
import os

from optimizer import optimize
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Self Configuring RAG",
    layout="wide"
)

st.title(
    "Self Configuring RAG Optimizer"
)

os.makedirs("uploads", exist_ok=True)
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    save_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(
        save_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.success(
        "PDF Uploaded Successfully"
    )

    if st.button(
        "Optimize RAG"
    ):

        with st.spinner(
            "Finding Best Parameters..."
        ):

            study = optimize(
                save_path
            )

        best = study.best_params

        st.subheader(
            "Best Parameters"
        )

        st.write(best)

        st.session_state.best = best
        st.session_state.pdf = save_path

if "best" in st.session_state:

    st.divider()

    st.subheader(
        "Ask Questions"
    )

    question = st.text_input(
        "Enter Question"
    )

    if st.button(
        "Get Answer"
    ):

        params = st.session_state.best

        rag = RAGPipeline(
            st.session_state.pdf,
            params["chunk_size"],
            params["chunk_overlap"],
            params["top_k"]
        )

        answer = rag.answer(
            question
        )

        st.write(answer)