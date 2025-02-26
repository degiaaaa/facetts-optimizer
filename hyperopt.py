import subprocess
import sys
import cluster_utils

def main(params):
    """
    Main function to run the hyperparameter optimization with cluster_utils.
    """
    job_id = params.get("id", "unknown")
    working_dir = params.get("working_dir", "./hp_results")
    print(f"[JOB {job_id}] Running hyperparameter optimization in {working_dir}...")

    try:
        # Start training with the given parameters
        train_result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
        print(train_result.stdout)
        if train_result.returncode != 0:
            print("[ERROR] Training failed.", file=sys.stderr)
            sys.exit(train_result.returncode)

        # Start evaluation
        eval_result = subprocess.run(["python", "evaluation/eval.py"], capture_output=True, text=True)
        print(eval_result.stdout)
        if eval_result.returncode != 0:
            print("[ERROR] Evaluation failed.", file=sys.stderr)
            sys.exit(eval_result.returncode)

        # Extract relevant metric
        composite_metric = extract_composite_metric(eval_result.stdout)
        print(f"[JOB {job_id}] Composite Metric: {composite_metric}")

        return {"composite_metric": composite_metric}

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}", file=sys.stderr)
        sys.exit(1)

def extract_composite_metric(output):
    """ Extracts the composite metric from evaluation output """
    import re
    match = re.search(r"Composite Metric:\s*([\d.-]+)", output)
    if match:
        return float(match.group(1))
    return None

if __name__ == "__main__":
    params = cluster_utils.initialize_job()
    results = main(params)
    cluster_utils.finalize_job(results)
