#!/bin/sh
# Final submission in one command: fleet blend + composition for US/India (final_assemble.sh), France from file 11
# (conflict rule + blend union at 0.98), a France blend union at THRESHOLD (optionally with swaps), and the
# address-less owner model at MIN_PM. Usage: ./scripts/final_build.sh NAME [THRESHOLD=0.9] [SWAPS=1] [MIN_PM=0.9]
set -e
NAME=${1:?name}; THR=${2:-0.9}; SWAPS=${3:-1}; MINPM=${4:-0.9}
PY=.venv/bin/python
./scripts/final_assemble.sh "$NAME" output/final_pre1_frcu/matching_results.tsv
mkdir -p "output/final_${NAME}_u" "output/final_${NAME}_own"
FLAG=""; [ "$SWAPS" = 1 ] && FLAG="--with-swaps"
$PY scripts/france_union.py --base "output/final_$NAME/matching_results.tsv" --scores "output/blend_$NAME/test_scored.parquet" \
    --output "output/final_${NAME}_u/matching_results.tsv" --threshold "$THR" $FLAG
$PY scripts/noaddr_owner.py "output/final_${NAME}_u/matching_results.tsv"
$PY scripts/add_owner.py --base "output/final_${NAME}_u/matching_results.tsv" --output "output/final_${NAME}_own/matching_results.tsv" --min-pm "$MINPM"
ln -f "output/final_$NAME/candidate_pairs.tsv" "output/final_${NAME}_own/candidate_pairs.tsv"
$PY src/utils_validate_submission.py -m "output/final_${NAME}_own/matching_results.tsv" -c "output/final_${NAME}_own/candidate_pairs.tsv" | tail -1
echo "done output/final_${NAME}_own"
