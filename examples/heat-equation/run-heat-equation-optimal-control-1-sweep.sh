#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXAMPLE_SCRIPT="${SCRIPT_DIR}/heat-equation-optimal-control-1.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/sweep-output}"
MAX_JOBS="${MAX_JOBS:-4}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/all-experiments-final-errors.csv}"
FINAL_TIME=0.002

WINDOW_SIZES=(2 3 4 5 6 8)
WINDOW_STEPS=(1 2 3 4 5)
DECAY_CONSTANTS=(0.01 0.1 1.0 10.0 100.0)
MISFIT_WEIGHTS=(0.01 0.1 1.0 10.0 100.0)

mkdir -p "${OUTPUT_ROOT}"
rm -f "${SUMMARY_CSV}"

safe_name() {
    printf '%s' "$1" | tr '.-' '__'
}

wait_for_one_job() {
    local pid

    while true; do
        for pid in $(jobs -p); do
            if wait "${pid}"; then
                return 0
            fi
            return 1
        done
        sleep 0.1
    done
}

active_jobs=0
failed_jobs=0

for window_size in "${WINDOW_SIZES[@]}"; do
    for window_step in "${WINDOW_STEPS[@]}"; do
        if [ "${window_size}" -le "${window_step}" ]; then
            continue
        fi

        for decay_constant in "${DECAY_CONSTANTS[@]}"; do
            for misfit_weight in "${MISFIT_WEIGHTS[@]}"; do
                run_name="window_size_${window_size}__window_step_${window_step}__decay_constant_$(safe_name "${decay_constant}")__misfit_weight_$(safe_name "${misfit_weight}")"
                outfile_path="${OUTPUT_ROOT}/${run_name}"

                cmd=(
                    "${PYTHON_BIN}" "${EXAMPLE_SCRIPT}"
                    --final-time "${FINAL_TIME}"
                    --decay-constant "${decay_constant}"
                    --misfit-weight "${misfit_weight}"
                    --window-size "${window_size}"
                    --window-step "${window_step}"
                    --outfile-path "${outfile_path}"
                    --summary-csv-path "${SUMMARY_CSV}"
                    --pvd-output False
                )

                mkdir -p "${outfile_path}"
                printf 'Launching window_size=%s window_step=%s decay_constant=%s misfit_weight=%s\n' "${window_size}" "${window_step}" "${decay_constant}" "${misfit_weight}"
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

while [ "${active_jobs}" -gt 0 ]; do
    if ! wait_for_one_job; then
        failed_jobs=1
    fi
    active_jobs=$((active_jobs - 1))
done

exit "${failed_jobs}"
