learning_rates=(1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 9e-5 10e-5 11e-5 12e-5 15e-5 2e-4 3e-4)

for lr in "${learning_rates[@]}"
do
    ./run_parallel.sh --dataset SocialHistory --model answerdotai/ModernBERT-base --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model /home/ubuntu/hf_models_new/phase2_clinical_3ep_decay_after_3ep_15mr/modernbert_phase2_clinical_3ep_decay_after_3ep_15mr --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model /home/ubuntu/hf_models_new/phase2_clinical_2ep_constant_1ep_decay_after_3ep_15mr/modernbert_phase2_clinical_2ep_constant_1ep_decay_after_3ep_15mr --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model /home/ubuntu/hf_models_new/phase2_bio_1ep_decay_after_3ep_15mr/modernbert_phase2_bio_1ep_decay_after_3ep_15mr --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model emilyalsentzer/Bio_ClinicalBERT --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model allenai/biomed_roberta_base --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model dmis-lab/biobert-v1.1 --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model yikuan8/Clinical-BigBird --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model yikuan8/Clinical-Longformer --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model Simonlee711/Clinical_ModernBERT --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model answerdotai/ModernBERT-large --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model /home/ubuntu/hf_models_new/phase2_large_clinical_2ep_constant_1ep_decay_after_3ep_15mr/modernbert_phase2_large_clinical_2ep_constant_1ep_decay_after_3ep_15mr --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
    ./run_parallel.sh --dataset SocialHistory --model /home/ubuntu/hf_models_new/phase2_large_clinical_3ep_decay_after_3ep_15mr/modernbert_phase2_large_clinical_3ep_decay_after_3ep_15mr --lr $lr --wd 1e-5 --epochs 10 --batch_size 16 --accumulation_steps 1
done