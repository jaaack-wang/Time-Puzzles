models=(
    gpt-4.1-2025-04-14
    gpt-4.1-nano-2025-04-14
    gpt-5-2025-08-07
    gpt-5-nano-2025-08-07
    openai/gpt-oss-20b
    deepseek/deepseek-chat
    deepseek/deepseek-reasoner
    Qwen/Qwen3-30B-A3B-Thinking-2507
    Qwen/Qwen3-4B-Instruct-2507
    Qwen/Qwen3-4B-Thinking-2507
    Qwen/Qwen3-8B
    Qwen/Qwen3-14B
    mistralai/Ministral-3-8B-Instruct-2512
    mistralai/Ministral-3-8B-Reasoning-2512
)

conditions=(
    "no_tool_use"
)

num_test_samples=600
input_fp=data/puzzles.json
solution_counts_to_include="1 2 3 4 5 6"
cuda_visible_devices="0"

# Loop through each model
for condition in "${conditions[@]}"; do
    for model in "${models[@]}"; do

        if [ "$condition" = "no_tool_use" ]; then
            echo "Running experiment for model: $model with condition: ${condition}_with_explicit_constraints"

            python -m src.experiments.main \
                --input_fp $input_fp \
                --condition "$condition" \
                --model_name "$model" \
                --use_explicit_constraints \
                --test_run \
                --num_test_samples $num_test_samples \
                --solution_counts_to_include $solution_counts_to_include \
                --cuda_visible_devices $cuda_visible_devices

            echo "Finished model: $model with condition: ${condition}_with_explicit_constraints"
            echo "----------------------------------------"
        fi

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
