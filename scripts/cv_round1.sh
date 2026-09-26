#!/bin/sh
# Cross-validated comparison of blend feature sets (fold = ten hash buckets).
PY=.venv/bin/python
$PY scripts/blend_cv.py --extra-ce b=cache/ce_b --extra-ce c=cache/ce_c --extra-ce L=cache/ce_large --extra-ce base=cache/ce_base --tag genLB
$PY scripts/blend_cv.py --extra-ce b=cache/ce_b --extra-ce c=cache/ce_c --extra-ce L=cache/ce_large --extra-ce base=cache/ce_base --extra-ce basehard=cache/ce_base_hard --extra-ce L2=cache/ce_large2 --tag auto2
$PY scripts/blend_cv.py --extras name,full --extra-ce b=cache/ce_b --extra-ce c=cache/ce_c --extra-ce L=cache/ce_large --tag fullL
