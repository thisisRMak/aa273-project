#!/bin/bash

# Check if at least one argument (the command) was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# Create timestamps
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
START_TIME_READABLE=$(date +"%Y-%m-%d %H:%M:%S")
START_SECONDS=$(date +%s)
LOGFILE="log_${TIMESTAMP}.txt"

FULL_COMMAND="$*"

# 1. Write metadata to log and stdout
{
    echo "Command executed: $FULL_COMMAND"
    echo "Started at:       $START_TIME_READABLE"
    echo "------------------------------------------"
} >> "$LOGFILE"
echo "Saving Log file at:"
echo $LOGFILE

# 2. Execute the command

"$@" 2>&1 | tee -a "$LOGFILE"

# 3. Calculate Elapsed Time
END_SECONDS=$(date +%s)
END_TIME_READABLE=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED=$(( END_SECONDS - START_SECONDS ))

# 4. Final Readout
{
    echo "------------------------------------------"
    echo "Finished at:      $END_TIME_READABLE"
    echo "Total duration:   $ELAPSED seconds"
} >> "$LOGFILE"

echo "------------------------------------------" 
echo "Finished in $ELAPSED seconds."
echo "Output saved to:" 
echo $LOGFILE