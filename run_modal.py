'''

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml

'''

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["MODAL_ENVIRONMENT"] = "narly"
import sys
import modal
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, "/root/ai-toolkit")
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# define the volume for storing model outputs, using "creating volumes lazily": https://modal.com/docs/guide/volumes
# you will find your model, samples and optimizer stored in: https://modal.com/storage/your-username/main/flux-lora-models
model_volume = modal.Volume.from_name("goznak-lora-models", create_if_missing=True)

# modal_output, due to "cannot mount volume on non-empty path" requirement
MOUNT_DIR = "/root/ai-toolkit/modal_output"  # modal_output, due to "cannot mount volume on non-empty path" requirement
UPLOADED_DATASET_DIR = "/root/ai-toolkit/uploaded_datasets"

# define modal app
# Install packages in batches to avoid dependency resolution conflicts
# Core packages first, then others with pinned versions to reduce resolution complexity
image = (
    modal.Image.debian_slim(python_version="3.11")
    # install required system and pip packages, more about this modal approach: https://modal.com/docs/examples/dreambooth_app
    .apt_install("git", "libgl1", "libglib2.0-0")
    # Install core PyTorch packages first
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "torchao==0.10.0",
    )
    # Install core ML packages with pinned versions to avoid conflicts
    .pip_install(
        "transformers==4.57.3",
        "git+https://github.com/huggingface/diffusers@6bf668c4d217ebc96065e673d8a257fd79950d34",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "peft",
    )
    # Install image processing packages
    .pip_install(
        "opencv-python",
        "albumentations==1.4.15",
        "albucore==0.0.16",
        "kornia",
        "einops",
        "invisible-watermark",
    )
    # Install training-specific packages
    .pip_install(
        "lycoris-lora==1.8.3",
        "controlnet_aux==0.0.10",
        "git+https://github.com/jaretburkett/easy_dwpose.git",
        "k-diffusion",
        "open_clip_torch",
        "timm",
        "prodigyopt",
        "bitsandbytes",
        "optimum-quanto==0.2.4",
    )
    # Install utility packages
    .pip_install(
        "python-dotenv",
        "ftfy",
        "oyaml",
        "flatten_json",
        "pyyaml",
        "tensorboard",
        "toml",
        "pydantic",
        "omegaconf",
        "hf_transfer",
        "lpips",
        "pytorch_fid",
        "pytorch-wavelets==1.3.0",
        "sentencepiece",
    )
)

# mount for the entire ai-toolkit directory
# Get the directory where this script is located (ai-toolkit directory)
AI_TOOLKIT_DIR = os.path.dirname(os.path.abspath(__file__))
# example: "/Users/username/ai-toolkit" is the local directory, "/root/ai-toolkit" is the remote directory
image = image.add_local_dir(AI_TOOLKIT_DIR, remote_path="/root/ai-toolkit", ignore=[".git"])

# create the Modal app with the necessary mounts and volumes
app = modal.App(name="goznak-styles-ai-toolkit", image=image, volumes={MOUNT_DIR: model_volume})

# Check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # Set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)

import argparse

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


def _write_uploaded_dataset(dataset_files, job_identifier):
    dataset_root = os.path.join(UPLOADED_DATASET_DIR, job_identifier or "dataset")
    os.makedirs(dataset_root, exist_ok=True)

    for item in dataset_files:
        relative_path = item.get("relative_path")
        content = item.get("content")

        if not relative_path or content is None:
            continue

        destination = os.path.normpath(os.path.join(dataset_root, relative_path))
        if not destination.startswith(dataset_root):
            raise ValueError("Unsafe dataset path detected")

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as file_handle:
            file_handle.write(content)

    os.environ["UPLOADED_DATASET_ROOT"] = dataset_root
    return dataset_root


@app.function(
    # request a GPU with at least 24GB VRAM
    # more about modal GPU's: https://modal.com/docs/guide/gpu
    gpu="A100", # gpu="H100"
    # more about modal timeouts: https://modal.com/docs/guide/timeouts
    timeout=7200,  # 2 hours, increase or decrease if needed
    secrets = [modal.Secret.from_name("huggingface-secret")]
)
def main(
    config_file_list_str: str,
    recover: bool = False,
    name: str = None,
    job_id: str = None,
    style_id: str = None,
    dataset_files: list = None,
    checkpoint_upload_url: str = None,
    webhook_url: str = None,  # Kept for backward compatibility
):
    # Import here so it only runs in Modal container where oyaml is installed
    from toolkit.job import get_job
    
    # convert the config file list from a string to a list
    config_file_list = config_file_list_str.split(",")

    if job_id or style_id or dataset_files or checkpoint_upload_url or webhook_url:
        print("Received style training metadata:")
        print(f" - job_id: {job_id}")
        print(f" - style_id: {style_id}")
        print(f" - dataset files: {len(dataset_files) if dataset_files else 0}")
        if checkpoint_upload_url:
            print(f" - checkpoint_upload_url: {checkpoint_upload_url}")
        if webhook_url:
            print(f" - webhook_url: {webhook_url}")

    dataset_root = None
    if dataset_files:
        dataset_root = _write_uploaded_dataset(dataset_files, job_id or style_id)
        print(f"Uploaded dataset available at: {dataset_root}")

    jobs_completed = 0
    jobs_failed = 0

    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, name)
            
            job.config['process'][0]['training_folder'] = MOUNT_DIR
            os.makedirs(MOUNT_DIR, exist_ok=True)
            print(f"Training outputs will be saved to: {MOUNT_DIR}")
            
            # run the job
            job.run()
            
            # commit the volume after training
            model_volume.commit()
            
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print(f"Error running job: {e}")
            jobs_failed += 1
            if not recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser()

    # require at least one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if a job fails
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # optional name replacement for config file
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )

    # Style training parameters (for microservice integration)
    parser.add_argument(
        '--job-id',
        type=str,
        default=None,
        help='Job identifier from backend'
    )
    parser.add_argument(
        '--style-id',
        type=str,
        default=None,
        help='Style identifier'
    )
    parser.add_argument(
        '--dataset-files-json',
        type=str,
        default=None,
        help='Path to JSON file containing dataset files (list of {relative_path, content})'
    )
    parser.add_argument(
        '--checkpoint-upload-url',
        type=str,
        default=None,
        help='Presigned S3 URL for uploading checkpoint directly to S3'
    )
    parser.add_argument(
        '--webhook-url',
        type=str,
        default=None,
        help='Backend webhook URL to call when training completes (deprecated, use checkpoint_upload_url)'
    )
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Output modal_job_id as JSON for programmatic use'
    )

    args = parser.parse_args()

    # convert list of config files to a comma-separated string for Modal compatibility
    config_file_list_str = ",".join(args.config_file_list)

    # Load dataset files from JSON if provided
    dataset_files = None
    if args.dataset_files_json:
        with open(args.dataset_files_json, 'r', encoding='utf-8') as f:
            dataset_files_data = json.load(f)
            # Decode base64 content if needed
            import base64
            dataset_files = []
            for item in dataset_files_data:
                content = item.get('content')
                if isinstance(content, str):
                    # Assume base64 encoded
                    content = base64.b64decode(content)
                elif isinstance(content, list):
                    # Convert list of ints to bytes
                    content = bytes(content)
                dataset_files.append({
                    'relative_path': item['relative_path'],
                    'content': content
                })

    # Prepare call arguments
    call_kwargs = {
        'config_file_list_str': config_file_list_str,
        'recover': args.recover,
        'name': args.name or (f"style-{args.style_id}" if args.style_id else None),
        'job_id': args.job_id,
        'style_id': args.style_id,
        'dataset_files': dataset_files,
        'checkpoint_upload_url': args.checkpoint_upload_url or args.webhook_url,  # Use checkpoint_upload_url if provided, fallback to webhook_url for compatibility
        'webhook_url': args.webhook_url,  # Keep for backward compatibility
    }
    # Remove None values
    call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

    # For style training jobs (spawn mode), use the deployed function
    if args.output_json or args.job_id or args.style_id:
        # Reference the deployed function instead of using local app
        # This requires the app to be deployed first with: modal deploy run_modal.py
        app_name = app.name  # "goznak-styles-ai-toolkit"
        try:
            # Use from_name to reference the deployed function
            deployed_main = modal.Function.from_name(app_name, "main")
            result = deployed_main.spawn(**call_kwargs)
            modal_job_id = result.object_id if hasattr(result, 'object_id') else str(result)
            
            # Output JSON with modal_job_id
            if args.output_json:
                print(json.dumps({"modal_job_id": modal_job_id}))
        except Exception as e:
            print(f"Error: Failed to reference deployed function '{app_name}/main'.", file=sys.stderr)
            print(f"The Modal app should be automatically deployed on microservice startup.", file=sys.stderr)
            print(f"If this error persists, check the microservice logs for deployment errors.", file=sys.stderr)
            print(f"Original error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Original behavior: synchronous call using local app
        with app.run():
            main.call(**call_kwargs)
