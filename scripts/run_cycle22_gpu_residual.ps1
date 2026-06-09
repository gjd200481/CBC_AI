param(
    [int]$Epochs = 50,
    [int]$BatchSize = 64,
    [double]$LearningRate = 0.001,
    [int]$NumWorkers = 2
)

$ErrorActionPreference = "Stop"

python train\sweep_seven_beam_architecture.py `
    --models residual_cnn `
    --full-dataset `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --learning-rate $LearningRate `
    --device cuda `
    --num-workers $NumWorkers `
    --pin-memory `
    --experiment-tag cycle22_residual_full_${Epochs}epoch `
    --history-dir result\metrics\cycle22_residual_full_${Epochs}epoch `
    --summary-csv result\metrics\cycle22_residual_full_${Epochs}epoch_2026-06-09.csv `
    --figure-path result\figures\cycle22_residual_full_${Epochs}epoch_2026-06-09.png
