#!/usr/bin/env bash
# Merge per-job result and log parts (written by run_greedy_sk_dariah.pbs)
# into the final shared files. Safe to re-run any time (e.g. partway through
# a long batch, or once after it completes); it always rebuilds the final
# files fresh from whatever part files currently exist, and never deletes
# the parts themselves.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_PARTS_DIR="$SCRIPT_DIR/results_parts"
LOGS_PARTS_DIR="$SCRIPT_DIR/logs_parts"

# merge_parts GLOB HEADER OUT_FILE
# HEADER may be empty ("") to skip writing a header line.
merge_parts() {
    local glob_pattern="$1"
    local header="$2"
    local out_file="$3"

    shopt -s nullglob
    local parts=($glob_pattern)
    shopt -u nullglob

    if [ "${#parts[@]}" -eq 0 ]; then
        echo "no part files matching $glob_pattern; leaving $out_file untouched"
        return
    fi

    local sorted_parts
    IFS=$'\n' sorted_parts=($(printf '%s\n' "${parts[@]}" | sort -V))
    unset IFS

    local tmp="$out_file.tmp"
    if [ -n "$header" ]; then
        printf "%s\n" "$header" > "$tmp"
        cat "${sorted_parts[@]}" >> "$tmp"
    else
        cat "${sorted_parts[@]}" > "$tmp"
    fi
    mv "$tmp" "$out_file"
    echo "merged ${#sorted_parts[@]} part file(s) into $out_file"
}

for MODE in greedy random reluctant; do
    merge_parts \
        "$RESULTS_PARTS_DIR/results_${MODE}_*.txt" \
        "repeat N instance_seed run_seed min_energy_perspin elapsed_time" \
        "$SCRIPT_DIR/results_${MODE}.txt"
done

merge_parts "$LOGS_PARTS_DIR/stdout_*.log" "" "$SCRIPT_DIR/run_greedy_sk_dariah.out"
merge_parts "$LOGS_PARTS_DIR/stderr_*.log" "" "$SCRIPT_DIR/run_greedy_sk_dariah.err"
