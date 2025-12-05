'''

ostris/ai-toolkit on https://modal.com
Run training with the following command:
modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml

'''

import base64
import json
import os
import re
import shutil
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
modal_env = os.environ.get("MODAL_ENV") or os.environ.get("MODAL_ENVIRONMENT") or "main"
os.environ["MODAL_ENVIRONMENT"] = modal_env
import sys
import urllib.request
from pathlib import Path
from uuid import uuid4

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
APP_NAME = os.environ.get("MODAL_APP_NAME", "goznak-styles-ai-toolkit")
app = modal.App(name=APP_NAME, image=image, volumes={MOUNT_DIR: model_volume})

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


def _write_uploaded_dataset(dataset_files, dataset_root: str, config_root: str):
    os.makedirs(dataset_root, exist_ok=True)
    os.makedirs(config_root, exist_ok=True)
    
    for item in dataset_files:
        relative_path = item.get("relative_path")
        content = item.get("content")

        if not relative_path or content is None:
            continue

        # Decode base64 content if it's a string
        if isinstance(content, str):
            content = base64.b64decode(content)
        elif isinstance(content, list):
            # Convert list of ints to bytes
            content = bytes(content)

        # Handle config files specially - they have __config__/ prefix
        if relative_path.startswith("__config__/"):
            # Extract filename and write to config directory
            config_filename = relative_path[len("__config__/"):]
            destination = os.path.normpath(os.path.join(config_root, config_filename))
            if not destination.startswith(config_root):
                raise ValueError("Unsafe config path detected")
            print(f"Writing config file to: {destination}")
            with open(destination, "wb") as file_handle:
                file_handle.write(content)
        else:
            # Regular dataset file
            destination = os.path.normpath(os.path.join(dataset_root, relative_path))
            if not destination.startswith(dataset_root):
                raise ValueError("Unsafe dataset path detected")

            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as file_handle:
                file_handle.write(content)

    return dataset_root


def _load_dataset_from_dir(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dataset_files = []
    for file_path in dataset_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(dataset_path).as_posix()
        with open(file_path, "rb") as handle:
            content = base64.b64encode(handle.read()).decode("utf-8")
        dataset_files.append({"relative_path": relative_path, "content": content})

    return dataset_files


def _find_latest_checkpoint(root_dir: str):
    latest_path = None
    latest_mtime = -1
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".safetensors"):
                continue
            full_path = os.path.join(dirpath, filename)
            mtime = os.path.getmtime(full_path)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = full_path
    return latest_path


def _upload_checkpoint_file(checkpoint_path: str, upload_url: str, headers: dict | None = None):
    headers = headers or {}
    print(f"Uploading checkpoint {checkpoint_path} -> {upload_url}")
    with open(checkpoint_path, "rb") as handle:
        data = handle.read()

    request = urllib.request.Request(upload_url, data=data, method="PUT")
    for key, value in headers.items():
        request.add_header(key, value)
    request.add_header("Content-Length", str(len(data)))

    with urllib.request.urlopen(request, timeout=600) as response:
        status_code = response.getcode()
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(f"Checkpoint upload failed with status {status_code}")
        print(f"Checkpoint upload completed with status {status_code}")


def _maybe_upload_checkpoint(
    checkpoint_upload_url: str,
    checkpoint_headers: dict | None,
    checkpoint_s3_key: str | None,
    checkpoint_root: str,
):
    if not checkpoint_upload_url:
        return

    checkpoint_path = _find_latest_checkpoint(checkpoint_root)
    if not checkpoint_path:
        raise RuntimeError("Training completed but no checkpoint was found to upload")

    print(f"Found checkpoint {checkpoint_path} (s3_key={checkpoint_s3_key})")
    _upload_checkpoint_file(checkpoint_path, checkpoint_upload_url, checkpoint_headers)


def _sanitize_identifier(value: str | None) -> str:
    if not value:
        return ""
    sanitized = re.sub(r'[^a-zA-Z0-9._-]+', '-', value.strip())
    return sanitized.strip("-")


def _generate_job_identifier(job_id: str | None, style_id: str | None, name: str | None) -> str:
    for candidate in (job_id, style_id, name):
        identifier = _sanitize_identifier(candidate)
        if identifier:
            return identifier
    return f"job-{uuid4().hex}"


def _prepare_job_paths(job_id: str | None, style_id: str | None, name: str | None):
    identifier = _generate_job_identifier(job_id, style_id, name)
    job_root = os.path.join(MOUNT_DIR, identifier)
    dataset_root = os.path.join(job_root, "dataset")
    config_root = os.path.join(job_root, "config")
    os.makedirs(dataset_root, exist_ok=True)
    os.makedirs(config_root, exist_ok=True)
    return {
        "identifier": identifier,
        "root": job_root,
        "dataset_dir": dataset_root,
        "config_dir": config_root,
    }


def _cleanup_job_directory(job_root: str):
    if not job_root:
        return
    mount_root = os.path.abspath(MOUNT_DIR)
    target_root = os.path.abspath(job_root)
    if not target_root.startswith(mount_root):
        print(f"Skipping cleanup for unexpected path outside mount: {target_root}")
        return
    if os.path.exists(target_root):
        print(f"Cleaning up job directory: {target_root}")
        shutil.rmtree(target_root, ignore_errors=True)


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
    checkpoint_headers: dict = None,
    checkpoint_s3_key: str = None,
    webhook_url: str = None,  # Kept for backward compatibility
):
    # Import here so it only runs in Modal container where oyaml is installed
    from toolkit.job import get_job
    from toolkit.config import get_config
    
    # convert the config file list from a string to a list
    config_file_list = config_file_list_str.split(",")
    job_paths = _prepare_job_paths(job_id, style_id, name)
    job_root = job_paths["root"]
    dataset_dir = job_paths["dataset_dir"]
    config_dir_path = Path(job_paths["config_dir"])
    print(f"Job artifacts directory: {job_root}")

    if job_id or style_id or dataset_files or checkpoint_upload_url or webhook_url:
        print("Received style training metadata:")
        print(f" - job_id: {job_id}")
        print(f" - style_id: {style_id}")
        print(f" - dataset files: {len(dataset_files) if dataset_files else 0}")
        if checkpoint_upload_url:
            print(f" - checkpoint_upload_url: {checkpoint_upload_url}")
        if checkpoint_s3_key:
            print(f" - checkpoint_s3_key: {checkpoint_s3_key}")
        if webhook_url:
            print(f" - webhook_url: {webhook_url}")

    dataset_root = None
    if dataset_files:
        dataset_root = _write_uploaded_dataset(dataset_files, dataset_dir, job_paths["config_dir"])
        print(f"Uploaded dataset available at: {dataset_root}")

    jobs_completed = 0
    jobs_failed = 0

    print(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            # Load config first so we can modify it before creating the job
            config_lookup = config_file
            if config_dir_path.exists():
                override_path = config_dir_path / Path(config_file).name
                if override_path.exists():
                    config_lookup = str(override_path)
                    if config_lookup != config_file:
                        print(f"Resolved uploaded config to: {config_lookup}")
            config = get_config(config_lookup, name)
            
            # Update dataset paths if we have an uploaded dataset (before job creation)
            if dataset_root:
                print(f"Updating dataset paths to use uploaded dataset: {dataset_root}")
                # Config structure: config['config']['process'][...]['datasets'][...]
                config_section = config.get('config', {})
                for process in config_section.get('process', []):
                    if 'datasets' in process:
                        for dataset in process['datasets']:
                            # Update folder_path if it exists
                            if 'folder_path' in dataset:
                                old_path = dataset['folder_path']
                                dataset['folder_path'] = dataset_root
                                print(f"  Updated dataset folder_path: {old_path} -> {dataset_root}")
                            # Update dataset_path if it exists (takes precedence over folder_path)
                            if 'dataset_path' in dataset:
                                old_path = dataset['dataset_path']
                                dataset['dataset_path'] = dataset_root
                                print(f"  Updated dataset dataset_path: {old_path} -> {dataset_root}")

            # Ensure every process writes checkpoints into the job-scoped directory so artifacts stay grouped
            config_section = config.get('config', {})
            processes = config_section.get('process', []) or []
            for process in processes:
                if isinstance(process, dict):
                    original_path = process.get('training_folder')
                    if original_path != job_root:
                        process['training_folder'] = job_root
                        if original_path:
                            print(f"Updated process training folder: {original_path} -> {job_root}")
            original_job_training_folder = config_section.get('training_folder')
            if original_job_training_folder != job_root:
                config_section['training_folder'] = job_root
                if original_job_training_folder:
                    print(f"Updated job training folder: {original_job_training_folder} -> {job_root}")
            
            # Create job with modified config
            job = get_job(config, name)
            
            print(f"Training outputs will be saved to: {job_root}")
            
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

    should_cleanup = bool(checkpoint_upload_url)
    try:
        _maybe_upload_checkpoint(checkpoint_upload_url, checkpoint_headers, checkpoint_s3_key, job_root)
    except Exception as upload_error:
        print(f"Failed to upload checkpoint: {upload_error}")
        raise
    else:
        if should_cleanup:
            # _cleanup_job_directory(job_root)
            pass

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
        '--dataset-dir',
        type=str,
        default=None,
        help='Path to directory containing extracted dataset files'
    )
    parser.add_argument(
        '--checkpoint-upload-url',
        type=str,
        default=None,
        help='Presigned S3 URL for uploading checkpoint directly to S3'
    )
    parser.add_argument(
        '--checkpoint-put-url',
        type=str,
        default=None,
        help='Presigned S3 PUT URL for uploading checkpoint bytes'
    )
    parser.add_argument(
        '--checkpoint-headers-json',
        type=str,
        default=None,
        help='JSON encoded headers required when uploading checkpoint'
    )
    parser.add_argument(
        '--checkpoint-s3-key',
        type=str,
        default=None,
        help='S3 key where the checkpoint will live'
    )
    parser.add_argument(
        '--webhook-url',
        type=str,
        default=None,
        help='Backend webhook URL to call when training completes (deprecated, use checkpoint_upload_url)'
    )
    parser.add_argument(
        '--modal-app-name',
        type=str,
        default=None,
        help='Override Modal app name when referencing deployed function'
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
    elif args.dataset_dir:
        dataset_files = _load_dataset_from_dir(args.dataset_dir)

    checkpoint_headers = {}
    if args.checkpoint_headers_json:
        checkpoint_headers = json.loads(args.checkpoint_headers_json)

    checkpoint_upload_url = args.checkpoint_put_url or args.checkpoint_upload_url or args.webhook_url

    # Prepare call arguments
    call_kwargs = {
        'config_file_list_str': config_file_list_str,
        'recover': args.recover,
        'name': args.name or (f"style-{args.style_id}" if args.style_id else None),
        'job_id': args.job_id,
        'style_id': args.style_id,
        'dataset_files': dataset_files,
        'checkpoint_upload_url': checkpoint_upload_url,
        'checkpoint_headers': checkpoint_headers or None,
        'checkpoint_s3_key': args.checkpoint_s3_key,
        'webhook_url': args.webhook_url,  # Keep for backward compatibility
    }
    # Remove None values
    call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}

    # For style training jobs (spawn mode), use the deployed function
    if args.output_json or args.job_id or args.style_id:
        # Reference the deployed function instead of using local app
        # This requires the app to be deployed first with: modal deploy run_modal.py
        app_name = args.modal_app_name or APP_NAME
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
