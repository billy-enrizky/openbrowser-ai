"""Start a vLLM server for the finetuned model.

Serves the merged (or base + LoRA) model via an OpenAI-compatible API.
Can be used with the eval pipeline via --models vllm://localhost:8000/model.

Usage:
    uv run infra/training/serving/serve_vllm.py --model outputs/serving/merged
    uv run infra/training/serving/serve_vllm.py --model Qwen/Qwen3-8B --lora-adapter outputs/finetuning_sft/final
    uv run infra/training/serving/serve_vllm.py --model Qwen/Qwen3-8B-AWQ --quantization awq
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def serve(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    lora_adapter: str | None = None,
    quantization: str | None = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
):
    """Start vLLM OpenAI-compatible API server."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
    ]

    if lora_adapter:
        cmd.extend(["--enable-lora", "--lora-modules", f"finetuned={lora_adapter}"])

    if quantization:
        cmd.extend(["--quantization", quantization])

    logger.info(f"Starting vLLM server: {' '.join(cmd)}")
    logger.info(f"API will be available at http://{host}:{port}/v1")
    logger.info(f"Use with eval: --models vllm://localhost:{port}/{model}")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error(
            "vLLM not installed. Install with: pip install vllm"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"vLLM server failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Start vLLM server for finetuned model"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B-AWQ",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--lora-adapter",
        help="Path to LoRA adapter (enables LoRA serving without merging)",
    )
    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method (use 'awq' for Qwen3-8B-AWQ)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max sequence length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )
    args = parser.parse_args()

    serve(
        model=args.model,
        host=args.host,
        port=args.port,
        lora_adapter=args.lora_adapter,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
