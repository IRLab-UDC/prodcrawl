#!/bin/bash
# Run one experiment configuration: every seed of its collection, one crawler
# process per site.
#
# Usage: ./experiments/launch_crawling_experiment.sh --config <config.yaml> [--commit-hash <hash>]
#
# The commit hash becomes part of the name of the Elasticsearch index that stores
# the run. It defaults to the current git commit of this repository.

CONFIG_PATH=""
COMMIT_HASH=""
while [[ $# -gt 0 ]]; do
	case $1 in
		--config) CONFIG_PATH="$2"; shift 2 ;;
		--commit-hash) COMMIT_HASH="$2"; shift 2 ;;
		*) shift ;;
	esac
done

if [ -z "$CONFIG_PATH" ]; then
	echo "ERROR: --config is required"
	exit 1
fi

cd "$(dirname "$0")/.."

if [ -z "$COMMIT_HASH" ]; then
	COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null)
fi
if ! [[ "$COMMIT_HASH" =~ ^[0-9a-f]+$ ]]; then
	echo "ERROR: run from a git checkout or pass --commit-hash <hex hash>"
	exit 1
fi

python3 -m experiments.run_spiders_runner --config "$CONFIG_PATH" --commit-hash "$COMMIT_HASH"
