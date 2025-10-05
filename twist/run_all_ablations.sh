#!/bin/bash
# Run twist ablation experiments for all baselines (twist_001 to twist_005)
#
# This script:
# 1. Takes each baseline twist (001-005)
# 2. Applies its reasoning to all other twists
# 3. Tests both think modes (no_more_thinking, allow_more_thinking)
# 4. Generates 1 rollout per condition
#
# Output structure:
#   ablation_outputs/baseline_twist_XXX/target_twist_YYY/think_mode/rollout_000.json

set -e  # Exit on error

# Configuration
NUM_ROLLOUTS=1
STORIES_DIR="/mnt/d/code/open_source/mats/thought_anchors_writing/data/twist/stories"

echo "================================================================================"
echo "Twist Ablation Experiments - Running All Baselines"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Baselines: twist_001 to twist_005"
echo "  Think modes: both (no_more_thinking + allow_more_thinking)"
echo "  Rollouts per condition: $NUM_ROLLOUTS"
echo "  Stories directory: $STORIES_DIR"
echo ""

# Check if stories directory exists
if [ ! -d "$STORIES_DIR" ]; then
    echo "ERROR: Stories directory not found: $STORIES_DIR"
    exit 1
fi

# Check if twist stories exist
for i in {1..5}; do
    twist_file="$STORIES_DIR/twist_00${i}.json"
    if [ ! -f "$twist_file" ]; then
        echo "ERROR: Baseline story not found: $twist_file"
        exit 1
    fi
done

echo "✓ All baseline stories found"
echo ""

# Run ablation for each baseline
for i in {1..5}; do
    baseline="twist_00${i}"

    echo "================================================================================"
    echo "Processing baseline: $baseline ($i/5)"
    echo "================================================================================"
    echo ""

    # Run twist_ablation.py
    python twist/twist_ablation.py \
        --baseline "$baseline" \
        --auto-find-twists \
        --num-rollouts "$NUM_ROLLOUTS" \
        --think-mode both \
        --stories-dir "$STORIES_DIR"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Completed: $baseline"
        echo ""
    else
        echo ""
        echo "✗ Failed: $baseline"
        echo ""
        exit 1
    fi
done

echo "================================================================================"
echo "All ablation experiments completed!"
echo "================================================================================"
echo ""
echo "Output location: /mnt/d/code/open_source/mats/thought_anchors_writing/data/twist/ablation_outputs/"
echo ""
echo "Directory structure:"
echo "  ablation_outputs/"
echo "    baseline_twist_001/"
echo "      target_twist_002/..."
echo "      target_twist_003/..."
echo "      ..."
echo "    baseline_twist_002/"
echo "      ..."
echo ""
echo "Next steps:"
echo "  1. Run semantic analysis: python twist/mechanistic_interp/analyze_all_ablations.py"
echo "  2. Run meta-analysis: python twist/mechanistic_interp/cross_twist_analysis.py"
echo ""
