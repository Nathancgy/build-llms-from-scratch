#!/bin/bash

# Create a timestamp for this run
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="logs_${timestamp}"

# Create log directory
mkdir -p $log_dir

echo "Starting training runs for all models at $(date)" | tee -a $log_dir/main.log

# Function to run training with timeout
run_model() {
    model_type=$1
    timeout_hours=$2
    epochs=$3
    
    echo "Starting $model_type model at $(date)" | tee -a $log_dir/main.log
    
    # Convert hours to seconds
    timeout_seconds=$((timeout_hours * 3600))
    timeout $timeout_seconds python train_unified.py \
        --model_type $model_type \
        --max_epochs $epochs \
        --eval_interval 100000000 \
        --save_interval 100000000 \
        --log_interval 10 \
        > $log_dir/${model_type}.log 2>&1
    
    exit_code=$?
    
    # Check if timeout occurred
    if [ $exit_code -eq 124 ]; then
        echo "WARNING: $model_type model timed out after $timeout_hours hours" | tee -a $log_dir/main.log
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR: $model_type model failed with exit code $exit_code" | tee -a $log_dir/main.log
    else
        echo "Finished $model_type model successfully at $(date)" | tee -a $log_dir/main.log
    fi
    
    # Kill any stray processes related to this model
    pkill -f "train_unified.py --model_type $model_type" || true
    
    # Add a delay to ensure clean shutdown
    sleep 10
    
    return $exit_code
}

# Try to run each model with a timeout
# If a model fails, continue to the next one

# Set epochs for full training runs
EPOCHS=15

# Train with each model
echo "Running all models with $EPOCHS epochs each" | tee -a $log_dir/main.log

# The order: Scale, Tauon, Muon
run_model "scale" 6 $EPOCHS
echo "Moving to next model..." | tee -a $log_dir/main.log

run_model "tauon" 6 $EPOCHS
echo "Moving to next model..." | tee -a $log_dir/main.log

run_model "muon" 6 $EPOCHS

echo "All training runs completed at $(date)" | tee -a $log_dir/main.log
echo "Log files are available in $log_dir" 