import subprocess
import os
import sys
import json
import re
from cluster_utils import read_params_from_cmdline, save_metrics_params

def run_command(command, working_dir, log_prefix):
    """
    Helper function to run a subprocess with logging.
    """
    print(f"[INFO] Running command: {' '.join(command)} in {working_dir}")

    stdout_log = os.path.join(working_dir, f"{log_prefix}_stdout.log")
    stderr_log = os.path.join(working_dir, f"{log_prefix}_stderr.log")

    result = subprocess.run(command, capture_output=True, text=True)

    with open(stdout_log, "w") as f:
        f.write(result.stdout)
    with open(stderr_log, "w") as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"[ERROR] {log_prefix} failed! Check logs.")
        sys.exit(result.returncode)

    return result.stdout

def extract_composite_metric(output):
    """
    Extracts the Composite Metric from the eval.py output.
    """
    match = re.search(r"Composite Metric:\\s*([\\d\\.-eE]+)", output)
    if match:
        return float(match.group(1))
    else:
        print("[ERROR] Composite Metric not found in evaluation output!")
        sys.exit(1)

def cluster_params_to_sacred(params):
    """
    Converts cluster-params to Sacred-compatible 'key=value' strings.
    """
    sacred_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                sacred_params.append(f"{key}.{sub_key}={sub_value}")
        else:
            sacred_params.append(f"{key}={value}")
    return sacred_params

def filter_optimized_params(params):
    """
    Only keep parameters that are used as hyperparameters.
    """
    allowed_keys = [
        "denoise_factor", "batch_size", "gamma", "disc_base_channels", "disc_num_layers",
        "disc_lrelu_slope", "disc_learning_rate", "lReLU_slope", "use_spectral_norm",
        "residual_channels", "warmup_disc_epochs", "freeze_gen_epochs", "disc_loss_type",
        "lambda_adv", "micro_batch_size"
    ]
    return {k: v for k, v in params.items() if k in allowed_keys}

if __name__ == "__main__":
    # 1. Read params from cluster_utils
    params = read_params_from_cmdline()
    job_id = params.get("id", "unknown")

    # ------------------------------------------------------------------
    # IMPORTANT CHANGE: Force the working directory to the project root.
    # This ensures any relative paths (e.g., "utils/cmu_dictionary")
    # are found under /mnt/qb/work/butz/bst080/faceGANtts.
    # ------------------------------------------------------------------
    os.chdir("/mnt/qb/work/butz/bst080/faceGANtts")

    # 2. Set working directory
    working_dir = params.get("working_dir", "/mnt/qb/work/butz/bst080/faceGANtts")
    os.makedirs(working_dir, exist_ok=True)

    # 3. Save JSON config
    param_config_path = os.path.join(working_dir, f"params_{job_id}.json")
    with open(param_config_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"[INFO] Saved job parameters to {param_config_path}")

    # 4. Filter hyperparameters
    train_eval_params = filter_optimized_params(params)

    # 5. Train script
    train_script = "/mnt/qb/work/butz/bst080/faceGANtts/train.py"
    run_command(
        [
            "python", "-u", train_script, "main", "with"
        ] + cluster_params_to_sacred(train_eval_params),
        working_dir,
        log_prefix=f"train_{job_id}"
    )

    # 6. Eval script
    eval_script = "/mnt/qb/work/butz/bst080/faceGANtts/evaluation/eval.py"
    eval_output = run_command(
        [
            "python", "-u", eval_script, "main", "with"
        ] + cluster_params_to_sacred(train_eval_params),
        working_dir,
        log_prefix=f"eval_{job_id}"
    )

    # 7. Extract composite metric
    composite_metric = extract_composite_metric(eval_output)
    print(f"[RESULT] Composite Metric for Job {job_id}: {composite_metric}")

    # 8. Report back to cluster_utils
    metrics = {"result": composite_metric}
    save_metrics_params(metrics, params)
