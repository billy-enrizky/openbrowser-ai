.PHONY: help eval-local eval-docker eval-stress eval-mind2web eval-formfactory \
       eval-aws-launch eval-aws-results \
       download-datasets upload-datasets download-results report \
       train-submit-sft train-submit-grpo train-submit-flow \
       train-submit-flow-llm-sft \
       train-submit-online-flow-grpo train-submit-online-flow-llm-grpo \
       train-submit-online-grpo \
       train-submit-fsdfm-sft train-submit-online-fsdfm-grpo \
       train-submit-eval-sft \
       train-list-jobs \
       preprocess-mind2web preprocess-formfactory \
       serve-vllm serve-ollama export-gguf \
       docker-build-eval docker-run-eval \
       terraform-plan-eval terraform-apply-eval terraform-destroy-eval \
       clean

# Configuration
EVAL_DATASETS_BUCKET ?= openbrowser-eval-datasets
EVAL_RESULTS_BUCKET ?= openbrowser-eval-results

# Default target
help:
	@echo "OpenBrowser-AI Infrastructure Makefile"
	@echo ""
	@echo "Evaluation:"
	@echo "  eval-local          Run eval locally (stress_tests, 5 tasks)"
	@echo "  eval-stress         Run stress test evaluation"
	@echo "  eval-mind2web       Run Mind2Web evaluation"
	@echo "  eval-formfactory    Run FormFactory evaluation"
	@echo "  eval-docker         Run eval via Docker"
	@echo "  eval-aws-launch     Launch eval on AWS spot instance"
	@echo "  eval-aws-results    Download eval results from S3"
	@echo "  report              Generate evaluation report from latest results"
	@echo ""
	@echo "Data:"
	@echo "  download-datasets       Download datasets from official sources"
	@echo "  upload-datasets         Upload datasets to S3"
	@echo "  download-results        Download results from S3"
	@echo "  preprocess-mind2web     Preprocess Mind2Web data for training"
	@echo "  preprocess-formfactory  Preprocess FormFactory data for training"
	@echo ""
	@echo "Training (Anyscale):"
	@echo "  train-submit-sft              Submit finetuning SFT job (Qwen3-8B QLoRA)"
	@echo "  train-submit-grpo             Submit finetuning GRPO job"
	@echo "  train-submit-flow             Submit flow matching job"
	@echo "  train-submit-flow-llm-sft         Submit flow LLM SFT job (ReFusion 8B, MDM)"
	@echo "  train-submit-online-flow-grpo     Submit online flow GRPO job (30M model, browser)"
	@echo "  train-submit-online-flow-llm-grpo Submit online flow LLM GRPO job (8B model, browser)"
	@echo "  train-submit-online-grpo          Submit online AR GRPO job (browser execution)"
	@echo "  train-submit-fsdfm-sft            Submit FS-DFM SFT job (1.3B DFM, LoRA)"
	@echo "  train-submit-online-fsdfm-grpo    Submit FS-DFM online GRPO job (browser)"
	@echo "  train-submit-eval-sft             Submit SFT-only eval job (browser, no training)"
	@echo "  train-list-jobs               List available training jobs"
	@echo ""
	@echo "Serving:"
	@echo "  serve-vllm          Start vLLM server (Qwen3-8B-AWQ)"
	@echo "  serve-ollama        Create and run Ollama model from GGUF"
	@echo "  export-gguf         Merge LoRA + convert to GGUF for Ollama"
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

eval-formfactory:
	$(EVAL_CMD) --datasets formfactory --max-tasks 10

eval-docker:
	docker compose -f infra/eval/docker/docker-compose.eval.yml up --build

report:
	$(REPORT_CMD) --input-dir results --output-dir results

eval-aws-launch:
	bash infra/eval/scripts/launch_eval.sh

eval-aws-results:
	uv run infra/eval/scripts/download_results.py \
		--bucket $$(cd infra/eval/terraform && terraform output -raw results_bucket)

# -------------------------------------------------------------------
# Data management
# -------------------------------------------------------------------

download-datasets:
	uv run infra/eval/scripts/download_datasets.py --all

upload-datasets:
	uv run infra/eval/scripts/upload_datasets.py --bucket $(EVAL_DATASETS_BUCKET)

download-results:
	uv run infra/eval/scripts/download_results.py

preprocess-mind2web:
	uv run infra/training/shared/mind2web_preprocessor.py

preprocess-formfactory:
	uv run infra/training/shared/formfactory_preprocessor.py

# -------------------------------------------------------------------
# Training (Anyscale Ray)
# -------------------------------------------------------------------

SUBMIT_CMD = uv run infra/training/anyscale/submit_job.py

# Preprocessed data file targets (only regenerate if missing)
data/processed/formfactory_sft.jsonl:
	uv run infra/training/shared/formfactory_preprocessor.py

data/processed/mind2web_flow.jsonl:
	uv run infra/training/shared/mind2web_preprocessor.py

train-submit-sft: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) finetuning-sft

train-submit-grpo: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) finetuning-grpo

train-submit-flow: data/processed/mind2web_flow.jsonl
	$(SUBMIT_CMD) flow-matching

data/processed/formfactory_flow.jsonl:
	uv run infra/training/shared/formfactory_preprocessor.py

train-submit-flow-llm-sft: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) flow-llm-sft

train-submit-online-flow-grpo: data/processed/formfactory_flow.jsonl
	$(SUBMIT_CMD) online-flow-grpo

train-submit-online-flow-llm-grpo: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) online-flow-llm-grpo

train-submit-online-grpo: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) online-grpo

train-submit-fsdfm-sft: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) fsdfm-sft

train-submit-online-fsdfm-grpo: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) online-fsdfm-grpo

train-submit-eval-sft: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) eval-sft

train-submit-eval-grpo: data/processed/formfactory_sft.jsonl
	$(SUBMIT_CMD) eval-grpo

train-list-jobs:
	$(SUBMIT_CMD) --list

# -------------------------------------------------------------------
# Model Serving
# -------------------------------------------------------------------

SERVE_VLLM_CMD = uv run infra/training/serving/serve_vllm.py
EXPORT_GGUF_CMD = uv run infra/training/serving/export_gguf.py

serve-vllm:
	$(SERVE_VLLM_CMD) --model Qwen/Qwen3-8B-AWQ --quantization awq

serve-ollama:
	ollama create qwen3-8b-formfactory -f outputs/serving/Modelfile
	ollama run qwen3-8b-formfactory

export-gguf:
	$(EXPORT_GGUF_CMD) --adapter outputs/finetuning_sft/final --output-dir outputs/serving

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

docker-build-eval:
	docker build -f infra/eval/docker/Dockerfile.eval -t openbrowser-eval .

docker-run-eval:
	docker run --rm \
		--shm-size=2g \
		-v $(PWD)/results:/app/results \
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
	rm -rf results/
	rm -rf outputs/finetuning_*
	rm -rf outputs/flow_matching_*
	rm -rf outputs/serving/
