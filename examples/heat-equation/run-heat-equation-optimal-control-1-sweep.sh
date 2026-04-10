#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXAMPLE_SCRIPT="${SCRIPT_DIR}/heat-equation-optimal-control-1.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/sweep-output}"
MAX_JOBS="${MAX_JOBS:-4}"

WINDOW_SIZES=(2 3 4 5 6 8 10)
WINDOW_STEPS=(1 2 3 4 5)
LAMBDAS=(0.0 0.01 0.05 0.1 0.2 0.5 1.0 2.0)
BETAS=(0.01 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0)
GAMMAS=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0)

mkdir -p "${OUTPUT_ROOT}"

safe_name() {
    printf '%s' "$1" | tr '.-' '__'
}

wait_for_one_job() {
    if ! wait -n; then
        return 1
    fi
}

active_jobs=0
failed_jobs=0

for window_size in "${WINDOW_SIZES[@]}"; do
    for window_step in "${WINDOW_STEPS[@]}"; do
        if [ "${window_size}" -le "${window_step}" ]; then
            continue
        fi

        for lambda_t in "${LAMBDAS[@]}"; do
            for beta in "${BETAS[@]}"; do
                for gamma in "${GAMMAS[@]}"; do
                    run_name="window_size_${window_size}__window_step_${window_step}__lambda_$(safe_name "${lambda_t}")__beta_$(safe_name "${beta}")__gamma_$(safe_name "${gamma}")"
                    outfile_path="${OUTPUT_ROOT}/${run_name}"

                    cmd=(
                        "${PYTHON_BIN}" "${EXAMPLE_SCRIPT}"
                        --lambda "${lambda_t}"
                        --beta "${beta}"
                        --gamma "${gamma}"
                        --window-size "${window_size}"
                        --window-step "${window_step}"
                        --outfile-path "${outfile_path}"
                    )

                    mkdir -p "${outfile_path}"
                    printf 'Launching window_size=%s window_step=%s lambda=%s beta=%s gamma=%s\n' "${window_size}" "${window_step}" "${lambda_t}" "${beta}" "${gamma}"
                    (
                        "${cmd[@]}"
                    ) >"${outfile_path}/stdout.log" 2>"${outfile_path}/stderr.log" &

                    active_jobs=$((active_jobs + 1))
                    if [ "${active_jobs}" -ge "${MAX_JOBS}" ]; then
                        if ! wait_for_one_job; then
                            failed_jobs=1
                        fi
                        active_jobs=$((active_jobs - 1))
                    fi
                done
            done
        done
    done
done

while [ "${active_jobs}" -gt 0 ]; do
    if ! wait_for_one_job; then
        failed_jobs=1
    fi
    active_jobs=$((active_jobs - 1))
done

exit "${failed_jobs}"
