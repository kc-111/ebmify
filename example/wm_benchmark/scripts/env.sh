#!/usr/bin/env bash
# Source me before running any swm-touching script:
#   source example/wm_benchmark/scripts/env.sh
#
# Points stable-worldmodel's cache (datasets + checkpoints) at this example's
# data/ dir instead of the upstream default (~/.stable_worldmodel/), so
# everything stays under the gitignored data/ folder.
export STABLEWM_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
echo "STABLEWM_HOME=$STABLEWM_HOME"
