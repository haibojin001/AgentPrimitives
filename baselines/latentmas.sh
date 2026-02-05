#!/bin/bash
set -e

BASELINE_DIR="baselines/"
REPO_URL="https://github.com/Gen-Verse/LatentMAS.git"

echo "=== Installing LatentMAS baseline ==="

mkdir -p baselines

if [ -d "$BASELINE_DIR" ]; then
  echo "⚠️  LatentMAS already exists at $BASELINE_DIR"
  echo "Skip cloning."
  exit 0
fi
echo "Cloning LatentMAS from $REPO_URL ..."
git clone "$REPO_URL" "$BASELINE_DIR"

echo "✅ Done!"
echo "LatentMAS is located at: $BASELINE_DIR"
