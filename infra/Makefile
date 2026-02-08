.PHONY: help eval-local eval-docker eval-stress eval-mind2web \
       upload-datasets download-results report \
       train-submit-sft train-submit-grpo train-submit-flow train-list-jobs \
       preprocess-mind2web \
       docker-build-eval docker-run-eval \
       terraform-plan-eval terraform-apply-eval terraform-destroy-eval \
       clean

# Default target
help:
	@echo "OpenBrowser-AI Infrastructure Makefile"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval-local          Run eval locally (stress_tests, 5 tasks)"
	@echo "  eval-stress         Run stress test evaluation"
	@echo "  eval-mind2web       Run Mind2Web evaluation"
	@echo "  eval-docker         Run eval via Docker"
	@echo "  report              Generate evaluation report from latest results"
	@echo ""
	@echo "Data:"
	@echo "  upload-datasets     Upload datasets to S3"
	@echo "  download-results    Download results from S3"
	@echo "  preprocess-mind2web Preprocess Mind2Web data for training"
	@echo ""
	@echo "Training (Anyscale):"
	@echo "  train-submit-sft    Submit finetuning SFT job"
	@echo "  train-submit-grpo   Submit finetuning GRPO job"
	@echo "  train-submit-flow   Submit flow matching job"
	@echo "  train-list-jobs     List available training jobs"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build-eval   Build eval Docker image"
	@echo "  docker-run-eval     Run eval in Docker container"
	@echo ""
	@echo "Terraform:"
	@echo "  terraform-plan-eval   Plan eval infrastructure"
	@echo "  terraform-apply-eval  Apply eval infrastructure"
	@echo "  terraform-destroy-eval Destroy eval infrastructure"
	@echo ""
	@echo "Other:"
	@echo "  clean               Remove local output files"

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

EVAL_CMD = uv run infra/eval/pipelines/eval_benchmark.py
REPORT_CMD = uv run infra/eval/pipelines/eval_report.py

eval-local:
	$(EVAL_CMD) --datasets stress_tests --max-tasks 5

eval-stress:
	$(EVAL_CMD) --datasets stress_tests

eval-mind2web:
	$(EVAL_CMD) --datasets mind2web --max-tasks 20

eval-docker:
	docker compose -f infra/eval/docker/docker-compose.eval.yml up --build

report:
	$(REPORT_CMD) --input-dir outputs/eval --output-dir outputs/eval

# -------------------------------------------------------------------
# Data management
# -------------------------------------------------------------------

upload-datasets:
	uv run infra/eval/scripts/upload_datasets.py

download-results:
	uv run infra/eval/scripts/download_results.py

preprocess-mind2web:
	uv run -c "from infra.training.shared.mind2web_preprocessor import main; main()"

# -------------------------------------------------------------------
# Training (Anyscale Ray)
# -------------------------------------------------------------------

SUBMIT_CMD = uv run infra/training/anyscale/submit_job.py

train-submit-sft:
	$(SUBMIT_CMD) finetuning-sft

train-submit-grpo:
	$(SUBMIT_CMD) finetuning-grpo

train-submit-flow:
	$(SUBMIT_CMD) flow-matching

train-list-jobs:
	$(SUBMIT_CMD) --list

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

docker-build-eval:
	docker build -f infra/eval/docker/Dockerfile.eval -t openbrowser-eval .

docker-run-eval:
	docker run --rm \
		--shm-size=2g \
		-v $(PWD)/outputs/eval:/app/outputs/eval \
		--env-file .env \
		openbrowser-eval \
		--datasets stress_tests --max-tasks 5

# -------------------------------------------------------------------
# Terraform (eval)
# -------------------------------------------------------------------

TF_EVAL_DIR = infra/eval/terraform

terraform-plan-eval:
	cd $(TF_EVAL_DIR) && terraform init && terraform plan

terraform-apply-eval:
	cd $(TF_EVAL_DIR) && terraform init && terraform apply

terraform-destroy-eval:
	cd $(TF_EVAL_DIR) && terraform destroy

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------

clean:
	rm -rf outputs/eval
	rm -rf outputs/finetuning_*
	rm -rf outputs/flow_matching_*
