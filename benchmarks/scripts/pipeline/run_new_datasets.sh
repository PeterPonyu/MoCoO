#!/bin/bash
# Run training on all 5 new datasets sequentially
set -e
cd /home/zeyufu/Desktop/MoCoO

DATASETS=("lung" "setty" "retina" "teeth" "hepatoblastoma")

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Starting: $ds  ($(date))"
    echo "=========================================="
    .venv/bin/python benchmarks/scripts/pipeline/fig1_training_pipeline.py --dataset "$ds" 2>&1 | tee "benchmarks/results/training_${ds}.log"
    echo "  Finished: $ds  ($(date))"
    echo ""
done

echo "=========================================="
echo "  ALL DATASETS COMPLETE  ($(date))"
echo "=========================================="
