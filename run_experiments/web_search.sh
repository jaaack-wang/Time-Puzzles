# ============== run live web search with gpt models ==============
models=(
    gpt-4.1-2025-04-14
    gpt-5-2025-08-07
)

conditions=(
    with_web_search
)

num_test_samples=300
input_fp=data/puzzles.json
solution_counts_to_include="1 3 5"
cuda_visible_devices="0"


# Loop through each model
for condition in "${conditions[@]}"; do
    for model in "${models[@]}"; do

        echo "Running experiment for model: $model with condition: $condition"

        python -m src.experiments.main \
            --input_fp $input_fp \
            --condition "$condition" \
            --model_name "$model" \
            --test_run \
            --num_test_samples $num_test_samples \
            --solution_counts_to_include $solution_counts_to_include \
            --cuda_visible_devices $cuda_visible_devices

        echo "Finished model: $model with condition: $condition"
        echo "----------------------------------------"

        # extract the cited web search results 
        output_fp="outputs/puzzles/with_web_search/${model}.json"
        python -m src.data.generate_pseudo_gpt_web_search_context $output_fp
    done
done


# ============== run pseduo web search with other models ==============

models=(
    openai/gpt-oss-20b
    deepseek/deepseek-reasoner
    Qwen/Qwen3-14B
    mistralai/Ministral-3-8B-Reasoning-2512
)

conditions=(
    with_pseudo_gpt_web_search_context
)

# Loop through each model
for condition in "${conditions[@]}"; do
    for model in "${models[@]}"; do

        echo "Running experiment for model: $model with condition: $condition"

        python -m src.experiments.main \
            --input_fp $input_fp \
            --condition "$condition" \
            --model_name "$model" \
            --test_run \
            --num_test_samples $num_test_samples \
            --solution_counts_to_include $solution_counts_to_include \
            --cuda_visible_devices $cuda_visible_devices

        echo "Finished model: $model with condition: $condition"
        echo "----------------------------------------"

    done
done
