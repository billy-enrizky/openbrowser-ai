#!/bin/bash
# Check SFT job status and submit RL jobs when all SFT jobs complete.
#
# Usage:
#   bash infra/training/anyscale/check_and_submit_rl.sh          # check SFT status
#   bash infra/training/anyscale/check_and_submit_rl.sh --submit # submit RL jobs
#   bash infra/training/anyscale/check_and_submit_rl.sh --push   # push checkpoints to HF

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Load env vars
set -a
source .env 2>/dev/null || true
set +a

export ANYSCALE_CLI_TOKEN="${ANYSCALE_CLI_TOKEN:-}"

SFT_JOBS=(
  "prodjob_rtc9bt77b4z3x15i8afimh3p1g|FlowVFE 39M SFT"
  "prodjob_i5nwqlz4x8rz8n5ric49pqgzrw|FS-DFM 1.3B SFT"
  "prodjob_igfbyz5hubb2cmff9ttt8lwr6r|ReFusion 8B SFT"
)

check_jobs() {
  local -n jobs=$1
  local label=$2
  echo "=== $label ==="
  local all_success=true
  for entry in "${jobs[@]}"; do
    job_id="${entry%%|*}"
    job_name="${entry##*|}"
    status=$(anyscale job list 2>/dev/null | grep "$job_id" | awk '{for(i=1;i<=NF;i++) if($i ~ /^(PENDING|STARTING|RUNNING|SUCCESS|FAILED|OUT_OF_RETRIES|STOPPED)$/) print $i}' | head -1)
    [ -z "$status" ] && status="UNKNOWN"
    echo "  $job_name: $status"
    if [ "$status" != "SUCCESS" ]; then
      all_success=false
    fi
  done
  echo ""
  $all_success
}

case "${1:-check}" in
  --submit)
    if check_jobs SFT_JOBS "SFT Job Status"; then
      echo "All SFT jobs completed! Submitting RL jobs..."
      echo ""
      echo "1/3: FlowVFE 39M Flow-GRPO..."
      uv run infra/training/anyscale/submit_job.py online-flow-grpo
      echo ""
      echo "2/3: FS-DFM 1.3B Flow-GRPO (v13 DCE)..."
      uv run infra/training/anyscale/submit_job.py fsdfm-flow-grpo
      echo ""
      echo "3/3: ReFusion 8B ESPO (v19)..."
      uv run infra/training/anyscale/submit_job.py espo-refusion
      echo ""
      echo "All 3 RL jobs submitted!"
    else
      echo "Not all SFT jobs are done yet. Check again later."
      exit 1
    fi
    ;;
  --push)
    echo "Submitting HuggingFace push job..."
    uv run infra/training/anyscale/submit_job.py push-to-hf
    echo "Push job submitted! Monitor on Anyscale console."
    ;;
  *)
    check_jobs SFT_JOBS "SFT Job Status" || true
    echo "Run with --submit to submit RL jobs after SFT completes."
    echo "Run with --push to push all checkpoints to HuggingFace."
    ;;
esac
