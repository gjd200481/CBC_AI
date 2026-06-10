param(
    [int]$Epochs = 50,
    [int]$BatchSize = 64,
    [double]$LearningRate = 0.001,
    [int]$NumWorkers = 2,
    [int]$Seed = 20260612,
    [string]$PhaseLoss = "cyclic"
)

$ErrorActionPreference = "Stop"

$tag = "cycle26_xie_mobilenet_${PhaseLoss}_${Epochs}epoch"

python train\sweep_seven_beam_architecture.py `
    --models mobilenetv3_small `
    --full-dataset `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --learning-rate $LearningRate `
    --seed $Seed `
    --device cuda `
    --num-workers $NumWorkers `
    --pin-memory `
    --phase-loss $PhaseLoss `
    --experiment-tag $tag `
    --history-dir result\metrics\$tag `
    --summary-csv result\metrics\${tag}_2026-06-10.csv `
    --figure-path result\figures\${tag}_2026-06-10.png
