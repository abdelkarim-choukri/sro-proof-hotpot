import subprocess
import os

# Get the actual user directory
user_dir = os.getenv('USER')  # This will get the username of the current user
WORKBASE = f"/var/tmp/u24sf51014/sro"
PROJ = f"{WORKBASE}/work/sro-proof-hotpot"
OUT0 = f"{WORKBASE}/work/sro-proof-hotpot/data/mdr/runs/hotpot_val_K20__mdr__rerun_20260217_1510"
OUT50 = f"{WORKBASE}/work/sro-proof-hotpot/data/mdr/runs/hotpot_val_K50__mdr__rerun_20260217_1510"
GOLD = f"{PROJ}/data/hotpotqa/raw/hotpot_dev_distractor_v1.json"  # Corrected file path

# Create directories for logs if they don't exist
os.makedirs(f"{OUT50}/logs", exist_ok=True)

# Function to run shell commands and redirect output to both file and terminal
def run_command(command, output_file):
    with open(output_file, "w") as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        for line in process.stdout:
            # Write to both terminal and file
            print(line.decode('utf-8'), end='')
            f.write(line.decode('utf-8'))

        # Handle any errors
        stderr = process.stderr.read().decode('utf-8')
        if stderr:
            print(stderr, file=sys.stderr)

# Look for any file plausibly recording the retrieval invocation
find_command = f"find {OUT0} -maxdepth 3 -type f \( -iname '*cmd*' -o -iname '*command*' -o -iname '*.log' -o -iname '*.sh' -o -iname '*manifest*.json' \) -print | sort"
run_command(find_command, f"{OUT50}/logs/out0_candidate_command_files.txt")

# Grep for likely retrieval lines
grep_command = f"grep -RIn --binary-files=without-match -E '(multihop|dense[_-]?retrieval|retriev|faiss|candidate_chains|topk|hotpot)' {OUT0} | head -n 200"
run_command(grep_command, f"{OUT50}/logs/out0_command_grep_head.txt")