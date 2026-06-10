param(
    [int]$Epochs = 50,
    [int]$BatchSize = 64,
    [double]$LearningRate = 0.001,
    [int]$NumWorkers = 2,
    [int]$Seed = 20260612
)

$ErrorActionPreference = "Stop"

python train\sweep_seven_beam_architecture.py `
    --models residual_cnn `
    --full-dataset `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --learning-rate $LearningRate `
    --seed $Seed `
    --device cuda `
    --num-workers $NumWorkers `
    --pin-memory `
    --experiment-tag cycle23_residual_best_${Epochs}epoch `
    --history-dir result\metrics\cycle23_residual_best_${Epochs}epoch `
    --summary-csv result\metrics\cycle23_residual_best_${Epochs}epoch_2026-06-10.csv `
    --figure-path result\figures\cycle23_residual_best_${Epochs}epoch_2026-06-10.png
