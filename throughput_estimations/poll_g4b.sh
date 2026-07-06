#!/bin/bash
# Authoritative poller for the optg4b_* Round-1 sweep (jobs 2496906-2496913).
# Runs in the same context as the Bash tool, so its squeue/sacct reads are trustworthy.
# Writes a heartbeat + harvested results to STATUS; exits when all 8 are terminal.
set -uo pipefail
JOBS_CSV="2496906,2496907,2496908,2496909,2496910,2496911,2496912,2496913"
JIDS="2496906 2496907 2496908 2496909 2496910 2496911 2496912 2496913"
LOG=/users/jminder/repositories/model-raising-data/logs
STATUS=/users/jminder/repositories/model-raising-data/logs/poll_g4b.status
RESULTS=/users/jminder/repositories/model-raising-data/logs/poll_g4b.results
declare -A NAME=( [2496906]=tp2dp2_fp8kv [2496907]=tp4 [2496908]=tp4_fp8kv [2496909]=pp2tp2 \
                  [2496910]=pp4 [2496911]=tp2dp2_mml12288 [2496912]=tp2dp2_seqs256 [2496913]=tp2dp2_memutil95 )
: > "$RESULTS"
harvested=""
while true; do
  ts=$(date "+%H:%M:%S")
  inq=$(squeue --me -h -o "%i" 2>/dev/null | grep -Ec "2496906|2496907|2496908|2496909|2496910|2496911|2496912|2496913")
  states=$(sacct -j "$JOBS_CSV" -X -n --format=State 2>/dev/null)
  nrec=$(echo "$states" | grep -Ec "PENDING|RUNNING|COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF|NODE_FAIL")
  nterm=$(echo "$states" | grep -Ec "COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF|NODE_FAIL")
  nrun=$(echo "$states" | grep -Ec "RUNNING")
  # harvest any new result files
  for j in $JIDS; do
    nm=${NAME[$j]}
    f="$LOG/optg4b_${nm}/throughput.out"
    if [ -s "$f" ] && grep -q "Samples/sec" "$f" 2>/dev/null && [[ "$harvested" != *"$j"* ]]; then
      { echo "### $nm (job $j)"; grep -E "Samples/sec|Input tokens:|Output tokens:|GPU-hours|successful|Wall time" "$f"; echo; } >> "$RESULTS"
      harvested="$harvested $j"
    fi
  done
  nres=$(grep -c "^###" "$RESULTS" 2>/dev/null || echo 0)
  echo "[$ts] inq=$inq sacct_rec=$nrec term=$nterm run=$nrun harvested=$nres" > "$STATUS"
  # Done: all 8 records present AND all terminal AND queue empty
  if [ "$nrec" -ge 8 ] && [ "$nterm" -ge 8 ] && [ "$inq" -eq 0 ]; then
    echo "[$ts] ALL DONE: 8 terminal, $nres results harvested" >> "$STATUS"
    echo "ALLDONE" >> "$STATUS"
    break
  fi
  sleep 90
done
