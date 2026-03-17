#!/bin/bash

# Check if at least one argument (the command) was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# Create a sortable timestamp (YYYYMMDD_HHMMSS)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="log_${TIMESTAMP}.txt"

# Capture the full command line call
# "$*" joins all arguments into a single string
FULL_COMMAND="$*"

# 1. Write the command being run to the log first
echo "Command executed: $FULL_COMMAND" >> "$LOGFILE"
echo "------------------------------------------" >> "$LOGFILE"

# 2. Execute the command
# "$@" preserves individual arguments even if they contain spaces
# 2>&1 merges stderr into stdout
# tee -a appends to the log while showing output in terminal
"$@" 2>&1 | tee -a "$LOGFILE"

echo "------------------------------------------" 
echo "Output saved to: $LOGFILE"

