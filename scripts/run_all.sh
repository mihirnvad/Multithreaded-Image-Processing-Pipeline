#!/usr/bin/env bash
# End-to-end run: dataset -> benchmark -> annotated images -> charts.
# Used as the Docker entrypoint; also works from a local checkout after
# building (BINARY defaults to build/image_pipeline if that exists).
set -euo pipefail

cd "$(dirname "$0")/.."

BINARY="${BINARY:-$(command -v image_pipeline || echo build/image_pipeline)}"
DATA="${DATA:-data/test_set}"
RESULTS="${RESULTS:-results}"
ASSETS="${ASSETS:-assets}"
COUNT="${COUNT:-200}"
REPEATS="${REPEATS:-3}"

if [[ ! -f "$DATA/manifest.csv" ]]; then
    echo "==> generating $COUNT synthetic images in $DATA"
    python3 scripts/generate_synthetic_dataset.py --count "$COUNT" --output "$DATA"
fi

echo "==> benchmarking $BINARY"
python3 scripts/benchmark.py --binary "$BINARY" --data "$DATA" --results "$RESULTS" --repeats "$REPEATS"

echo "==> writing annotated images and feature maps"
"$BINARY" --input "$DATA" --output "$RESULTS/images" --feature-maps --metrics "$RESULTS/metrics_annotated.csv"

echo "==> rendering charts"
python3 scripts/visualize_metrics.py --results "$RESULTS" --images "$RESULTS/images" --data "$DATA" --out "$ASSETS"
