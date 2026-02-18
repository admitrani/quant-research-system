from orchestration.stages import (
    run_ingestion,
    run_silver_transformations,
    run_gold_transformations,
    run_model_stage
)

# Current execution mode: batch

def run_pipeline():
    try:
        print("Stage 1: Ingestion")
        run_ingestion()

        print("Stage 2: Silver transformations")
        run_silver_transformations()

        print("Stage 3: Gold transformations")
        run_gold_transformations()

        print("Stage 4: Model stage")
        run_model_stage()

        print("Pipeline finished successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()