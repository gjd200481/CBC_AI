param(
    [int]$Epochs = 50,
    [int]$BatchSize = 32,
    [double]$LearningRate = 0.001,
    [double]$LambdaPhy = 0.1,
    [int]$NumWorkers = 2,
    [int]$Seed = 20260612
)

$ErrorActionPreference = "Stop"

$tag = "cycle25_residual_physics_lambda_${LambdaPhy}_${Epochs}epoch".Replace(".", "p")

python train\train_seven_beam_physics_constrained_cnn.py `
    --model-name residual_cnn `
    --lambda-phy $LambdaPhy `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --learning-rate $LearningRate `
    --seed $Seed `
    --device cuda `
    --num-workers $NumWorkers `
    --model-path models\${tag}.pth `
    --metrics-path result\metrics\${tag}_2026-06-10.csv `
    --summary-path result\metrics\${tag}_summary_2026-06-10.csv `
    --figure-path result\figures\${tag}_2026-06-10.png `
    --no-plot
