import optuna

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from rag_pipeline import RAGPipeline
from evaluation_data import evaluation_data

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def semantic_similarity(
    generated,
    expected
):

    emb1 = model.encode(
        generated,
        convert_to_tensor=True
    )

    emb2 = model.encode(
        expected,
        convert_to_tensor=True
    )

    score = cos_sim(
        emb1,
        emb2
    )

    return float(score)


def objective(
    trial,
    pdf_path
):

    chunk_size = trial.suggest_categorical(
        "chunk_size",
        [500,1000,1500]
    )

    chunk_overlap = trial.suggest_categorical(
        "chunk_overlap",
        [50,100,150]
    )

    top_k = trial.suggest_categorical(
        "top_k",
        [2,3,5]
    )

    rag = RAGPipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        top_k
    )

    total_score = 0

    for item in evaluation_data:

        try:

            generated_answer = rag.answer(
                item["question"]
            )

            similarity = semantic_similarity(
                generated_answer,
                item["ground_truth"]
            )

            total_score += similarity

        except:

            total_score += 0

    average_score = (
        total_score /
        len(evaluation_data)
    )

    return average_score


def optimize(pdf_path):

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial:
        objective(
            trial,
            pdf_path
        ),
        n_trials=10
    )

    return study
