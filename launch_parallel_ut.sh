#!/bin/bash
# Run downstream evals for a research/universal_transformer/train.py checkpoint/artifact across 8 GPUs.
#
# Usage:
#   ./launch_parallel_ut.sh <checkpoint.pt> [tasks] [batch_size] [output_dir] [max_length]
#
# Example:
#   ./launch_parallel_ut.sh runs/my-ut-run/model.pt "hellaswag,arc_easy,sciq,piqa" 8 results_ut 2048

set -e

WORLD_SIZE=8
CHECKPOINT="$1"
TASKS="${2:-hellaswag,arc_easy,sciq,piqa,arc_challenge,winogrande,openbookqa,lambada_openai}"
BATCH_SIZE="${3:-8}"
OUTPUT_DIR="${4:-results_ut}"
MAX_LENGTH="${5:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <checkpoint.pt> [tasks] [batch_size] [output_dir] [max_length]"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/launch_${TIMESTAMP}.log"

{
echo "========================================"
echo "Slowrun UT Multi-GPU Evaluation"
echo "========================================"
echo "Checkpoint:  $CHECKPOINT"
echo "Tasks:       $TASKS"
echo "Batch size:  $BATCH_SIZE"
echo "Max length:  $MAX_LENGTH"
echo "Python:      $PYTHON_BIN"
echo "Output dir:  $OUTPUT_DIR"
echo "World size:  $WORLD_SIZE GPUs"
echo "Log file:    $LOG_FILE"
echo "========================================"
echo ""
} | tee -a "$LOG_FILE"

cleanup() {
    echo "" | tee -a "$LOG_FILE"
    echo "Caught interrupt, killing background processes..." | tee -a "$LOG_FILE"
    kill $(jobs -p) 2>/dev/null || true
    exit 1
}
trap cleanup INT TERM

echo "Launching $WORLD_SIZE processes..." | tee -a "$LOG_FILE"
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    OUTPUT_FILE="$OUTPUT_DIR/results_rank${rank}.json"
    RANK_LOG="$OUTPUT_DIR/rank${rank}.log"
    echo "  GPU $rank -> $OUTPUT_FILE" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$rank "$PYTHON_BIN" bench_ut.py \
        --rank $rank \
        --world_size $WORLD_SIZE \
        --checkpoint "$CHECKPOINT" \
        --tasks "$TASKS" \
        --batch_size $BATCH_SIZE \
        --max_length $MAX_LENGTH \
        --device "cuda:0" \
        --output "$OUTPUT_FILE" \
        > "$RANK_LOG" 2>&1 &
    eval "PID_$rank=$!"
done

echo "" | tee -a "$LOG_FILE"
echo "All processes launched. Tail with: tail -f $OUTPUT_DIR/rank*.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

FAILED=0
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
    eval "PID=\$PID_$rank"
    if wait $PID; then
        echo "[OK] Rank $rank completed" | tee -a "$LOG_FILE"
    else
        echo "[FAIL] Rank $rank failed (see $OUTPUT_DIR/rank${rank}.log)" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$LOG_FILE"
if [ $FAILED -eq 0 ]; then
    echo "All processes completed. Combining..." | tee -a "$LOG_FILE"
    "$PYTHON_BIN" bench_ut.py combine "$OUTPUT_DIR"/results_rank*.json \
        --output "$OUTPUT_DIR/results_combined.json" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Done. Combined results: $OUTPUT_DIR/results_combined.json" | tee -a "$LOG_FILE"
    exit 0
else
    echo "ERROR: $FAILED processes failed. Check $OUTPUT_DIR/rank*.log" | tee -a "$LOG_FILE"
    exit 1
fi
