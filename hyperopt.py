import subprocess
import sys
import os
import cluster_utils


def main(params):
    """
    Main function to run the hyperparameter optimization with cluster_utils.
    """
    job_id = params.get("id", "unknown")
    working_dir = params.get("working_dir", "./hp_results")
    print(f"[JOB {job_id}] Running hyperparameter optimization in {working_dir}...")

    try:
        # Starte das Training mit Debugging-Logs
        train_result = subprocess.run(
            ["python", "-u", "train.py"],  # -u für unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(train_result.stdout)
        print(train_result.stderr, file=sys.stderr)

        if train_result.returncode != 0:
            print("[ERROR] Training failed.", file=sys.stderr)
            sys.exit(train_result.returncode)

        # Starte die Evaluation
        eval_result = subprocess.run(
            ["python", "-u", "evaluation/eval.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(eval_result.stdout)
        print(eval_result.stderr, file=sys.stderr)

        if eval_result.returncode != 0:
            print("[ERROR] Evaluation failed.", file=sys.stderr)
            sys.exit(eval_result.returncode)

        # Extrahiere den Metrikwert
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
