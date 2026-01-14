models=(
    gpt-4.1-2025-04-14
    gpt-5-2025-08-07
)

conditions=(
    "with_code_interpreter"
    "with_pseudo_gpt_web_search_context_and_code_interpreter"
)

num_test_samples=100
input_fp=data/puzzles.json
solution_counts_to_include="1"

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
            --solution_counts_to_include $solution_counts_to_include

        echo "Finished model: $model with condition: $condition"
        echo "----------------------------------------"

        if [ "$condition" = "no_tool_use" ] || [ "$condition" = "with_code_interpreter" ]; then
            echo "Running experiment for model: $model with condition: ${condition}_with_explicit_constraints"

            python -m src.experiments.main \
                --input_fp $input_fp \
                --condition "$condition" \
                --model_name "$model" \
                --use_explicit_constraints \
                --test_run \
                --num_test_samples $num_test_samples \
                --solution_counts_to_include $solution_counts_to_include

            echo "Finished model: $model with condition: ${condition}_with_explicit_constraints"
            echo "----------------------------------------"
        fi
    done
done