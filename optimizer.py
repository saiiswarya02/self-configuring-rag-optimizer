import optuna
from rag_pipeline import RAGPipeline


evaluation_questions = [
    "What is machine learning?",
    "What is deep learning?",
    "What is RAG?"
]


def objective(trial, pdf_path):

    chunk_size = trial.suggest_categorical(
        "chunk_size",
        [500, 1000, 1500]
    )

    chunk_overlap = trial.suggest_categorical(
        "chunk_overlap",
        [50, 100, 150]
    )

    top_k = trial.suggest_categorical(
        "top_k",
        [2, 3, 5]
    )

    rag = RAGPipeline(
        pdf_path,
        chunk_size,
        chunk_overlap,
        top_k
    )

    total_score = 0

    for q in evaluation_questions:

        try:

            answer = rag.answer(q)

            score = min(
                len(answer) / 100,
                1.0
            )

            total_score += score

        except:

            total_score += 0

    return total_score / len(
        evaluation_questions
    )


def optimize(pdf_path):

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            pdf_path
        ),
        n_trials=10
    )

    return study