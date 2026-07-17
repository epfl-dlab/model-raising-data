#!/bin/bash
# Authoritative poller for Round-2 (g4opt_*) + probe jobs. nohup-launched.
set -uo pipefail
LOG=/users/jminder/repositories/model-raising-data/logs
STATUS=$LOG/poll_r2.status
RESULTS=$LOG/poll_r2.results
# jobid -> run_tag (result dir under logs/)
declare -A NAME=(
  [2496979]=probe_tp2dp2_fp8_mml12k
  [2496988]=r2_tp4_fp8_mml12k
  [2496990]=r2_tp4_fp8_gmu95
  [2496991]=r2_tp4_fp8_seqs512
  [2496992]=r2_tp4_fp8_seqs256
  [2496993]=r2_pp4
  [2497010]=dbg_tp4_fp8_mml12k
)
JIDS="2496979 2496988 2496990 2496991 2496992 2496993 2497010"
CSV="2496979,2496988,2496990,2496991,2496992,2496993,2497010"
: > "$RESULTS"
harvested=""
while true; do
  ts=$(date "+%H:%M:%S")
  states=$(sacct -j "$CSV" -X -n --format=JobID,State 2>/dev/null)
  nterm=$(echo "$states" | grep -Ec "COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF|NODE_FAIL")
  ncan=$(echo "$states" | grep -Ec "CANCELLED")
  nrec=$(echo "$states" | grep -Ec "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF|NODE_FAIL")
  for j in $JIDS; do
    nm=${NAME[$j]}; f="$LOG/$nm/throughput.out"
    if [ -s "$f" ] && grep -q "Samples/sec" "$f" 2>/dev/null && [[ "$harvested" != *"$j"* ]]; then
      { echo "### $nm (job $j)"; grep -E "Samples/sec|Output tokens:|GPU-hours|successful|Wall time" "$f"; echo; } >> "$RESULTS"
      harvested="$harvested $j"
    fi
  done
  nres=$(grep -c "^###" "$RESULTS" 2>/dev/null || echo 0)
  echo "[$ts] sacct_rec=$nrec term=$nterm cancelled=$ncan harvested=$nres" > "$STATUS"
  if [ "$nrec" -ge 7 ] && [ "$nterm" -ge 7 ]; then
    echo "[$ts] ALLDONE term=$nterm cancelled=$ncan harvested=$nres" >> "$STATUS"
    break
  fi
  sleep 60
done
