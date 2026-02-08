"""Export finetuned LoRA model to GGUF format for Ollama serving.

Steps:
1. Load base Qwen3-8B model (full precision)
2. Load and merge LoRA adapter from checkpoint
3. Save merged model
4. Convert to GGUF using llama.cpp's convert script
5. Generate Ollama Modelfile

Usage:
    uv run infra/training/serving/export_gguf.py --adapter outputs/finetuning_sft/final
    uv run infra/training/serving/export_gguf.py --adapter outputs/finetuning_sft/final --quantize q4_k_m
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "serving"
DEFAULT_MODELFILE_TEMPLATE = Path(__file__).parent / "Modelfile"


def merge_lora_adapter(adapter_path: str, output_dir: Path):
    """Load base model + LoRA adapter and save merged model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from infra.training.finetuning.config import BASE_MODEL_NAME

    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.float16,
        device_map="cpu",
    )

    logger.info(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging LoRA adapter into base model")
    model = model.merge_and_unload()

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {merged_dir}")
    model.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.save_pretrained(merged_dir)

    logger.info("Merge complete")
    return merged_dir


def convert_to_gguf(
    merged_dir: Path, output_dir: Path, quantize: str = "q4_k_m"
) -> Path:
    """Convert merged HuggingFace model to GGUF format.

    Requires llama.cpp to be installed (llama-quantize and
    convert_hf_to_gguf.py available on PATH or in llama.cpp clone).
    """
    gguf_fp16 = output_dir / "model-fp16.gguf"
    gguf_quantized = output_dir / f"qwen3-8b-formfactory-{quantize}.gguf"

    # Step 1: Convert to GGUF FP16
    logger.info("Converting to GGUF FP16...")
    convert_cmd = [
        sys.executable, "-m", "llama_cpp.convert",
        str(merged_dir),
        "--outfile", str(gguf_fp16),
        "--outtype", "f16",
    ]

    # Try llama-cpp-python's converter first, fall back to llama.cpp script
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("llama-cpp-python convert not available, trying convert_hf_to_gguf.py")
        convert_cmd = [
            sys.executable, "convert_hf_to_gguf.py",
            str(merged_dir),
            "--outfile", str(gguf_fp16),
            "--outtype", "f16",
        ]
        try:
            subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(
                f"GGUF conversion failed: {e}\n"
                "Install llama-cpp-python or clone llama.cpp and run "
                "convert_hf_to_gguf.py manually."
            )
            return gguf_fp16

    if not gguf_fp16.exists():
        logger.error(f"FP16 GGUF not created at {gguf_fp16}")
        return gguf_fp16

    # Step 2: Quantize
    if quantize != "f16":
        logger.info(f"Quantizing to {quantize}...")
        quantize_cmd = ["llama-quantize", str(gguf_fp16), str(gguf_quantized), quantize]
        try:
            subprocess.run(quantize_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Quantized GGUF saved to: {gguf_quantized}")
            return gguf_quantized
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(
                f"Quantization failed ({e}), using FP16 GGUF instead. "
                "Install llama.cpp for quantization support."
            )
            return gguf_fp16

    return gguf_fp16


def generate_modelfile(gguf_path: Path, output_dir: Path) -> Path:
    """Generate an Ollama Modelfile for the converted model."""
    modelfile_path = output_dir / "Modelfile"

    content = (
        f"FROM {gguf_path}\n"
        "\n"
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER stop <|im_end|>\n"
        "\n"
        'SYSTEM """You are a web browser automation agent. Given a task, '
        "produce a step-by-step action plan to complete it. "
        "Each step should be a specific browser action: navigate, click, type, "
        'select, or submit."""\n'
    )

    modelfile_path.write_text(content)
    logger.info(f"Modelfile written to: {modelfile_path}")
    return modelfile_path


def create_ollama_model(modelfile_path: Path, model_name: str = "qwen3-8b-formfactory"):
    """Register the model with Ollama."""
    logger.info(f"Creating Ollama model: {model_name}")
    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True,
        )
        logger.info(f"Ollama model '{model_name}' created successfully")
        logger.info(f"Test with: ollama run {model_name}")
    except FileNotFoundError:
        logger.error(
            "Ollama CLI not found. Install Ollama from https://ollama.com "
            f"then run: ollama create {model_name} -f {modelfile_path}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create Ollama model: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export finetuned model to GGUF for Ollama"
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to LoRA adapter checkpoint (e.g. outputs/finetuning_sft/final)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for GGUF and Modelfile",
    )
    parser.add_argument(
        "--quantize",
        default="q4_k_m",
        choices=["f16", "q4_0", "q4_k_m", "q5_k_m", "q8_0"],
        help="GGUF quantization type (default: q4_k_m)",
    )
    parser.add_argument(
        "--model-name",
        default="qwen3-8b-formfactory",
        help="Ollama model name (default: qwen3-8b-formfactory)",
    )
    parser.add_argument(
        "--skip-ollama-create",
        action="store_true",
        help="Skip creating the Ollama model (just export GGUF + Modelfile)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge LoRA
    merged_dir = merge_lora_adapter(args.adapter, output_dir)

    # Step 2: Convert to GGUF
    gguf_path = convert_to_gguf(merged_dir, output_dir, quantize=args.quantize)

    # Step 3: Generate Modelfile
    modelfile_path = generate_modelfile(gguf_path, output_dir)

    # Step 4: Create Ollama model
    if not args.skip_ollama_create:
        create_ollama_model(modelfile_path, args.model_name)

    logger.info("Export complete")
    logger.info(f"  GGUF:      {gguf_path}")
    logger.info(f"  Modelfile: {modelfile_path}")


if __name__ == "__main__":
    main()
