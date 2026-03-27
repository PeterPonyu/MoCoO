#!/bin/bash
# Run training on all 5 new datasets sequentially
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

cd "$ROOT_DIR"

DATASETS=("lung" "setty" "retina" "teeth" "hepatoblastoma")

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Starting: $ds  ($(date))"
    echo "=========================================="
    "$PYTHON_BIN" benchmarks/scripts/pipeline/fig1_training_pipeline.py --dataset "$ds" 2>&1 | tee "benchmarks/results/training_${ds}.log"
    echo "  Finished: $ds  ($(date))"
    echo ""
done

echo "=========================================="
echo "  ALL DATASETS COMPLETE  ($(date))"
echo "=========================================="
