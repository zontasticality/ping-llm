#!/usr/bin/env bash
# Download checkpoint from Modal volume to local machine
# Usage: bash scripts/download_checkpoint.sh [checkpoint_step]

CHECKPOINT_STEP=${1:-2000}
VOLUME_NAME=${MODAL_VOLUME:-ping-llm}

# Download to match the same structure as on Modal
MODAL_PATH="outputs/latency_network/full_run/full_run/checkpoints/${CHECKPOINT_STEP}"
LOCAL_PARENT="outputs/latency_network/full_run/full_run/checkpoints"
LOCAL_FULL_PATH="${LOCAL_PARENT}/${CHECKPOINT_STEP}"

echo "Downloading checkpoint ${CHECKPOINT_STEP} from Modal volume ${VOLUME_NAME}..."
echo "  Remote: ${MODAL_PATH}"
echo "  Local:  ${LOCAL_FULL_PATH}"

# Create parent directory if needed
mkdir -p "${LOCAL_PARENT}"

# Use modal volume get to download the checkpoint directory
# Note: destination should be the parent directory where the folder will be created
modal volume get "${VOLUME_NAME}" \
  "${MODAL_PATH}" \
  "${LOCAL_PARENT}/"

echo "✓ Done! Checkpoint downloaded to: ${LOCAL_FULL_PATH}"
