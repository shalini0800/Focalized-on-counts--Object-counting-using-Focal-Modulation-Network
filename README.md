# Focalized-on-counts-Object-counting-using-Focal-Modulation-Network
explored the new architecture called focal modulation and its effectiveness for generalized visual object counting tasks. 

1. Pretrain execution:
CUDA_VISIBLE_DEVICES=5 python  CounTR/FSC_pretrain.py      --epochs 500 --model 'mae_vit_base_patch16'   --warmup_epochs 10    --blr 1.5e-4 --weight_decay 0.05

2. Finetune_cross execution:
CUDA_VISIBLE_DEVICES=7 python -u CounTR/FSC_finetune_cross.py    --epochs 1000  --model 'mae_vit_base_patch16'    --blr 2e-4 --weight_decay 0.05

3. Few-Shot execution:
CUDA_VISIBLE_DEVICES=7 python -u CounTR/FSC_test_cross\(few-shot\).py  --epochs 1 --model 'mae_vit_base_patch16'

4. Zero-Shot execution:
CUDA_VISIBLE_DEVICES=5 python -u CounTR/FSC_test_cross\(zero-shot\).py  --epochs 1 --model 'mae_vit_base_patch16'
