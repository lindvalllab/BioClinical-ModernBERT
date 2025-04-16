# ./run_parallel.sh --dataset HOC --model answerdotai/ModernBERT-base --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1

# ./run_parallel.sh --dataset HOC --model /home/ubuntu/hf_models_new/phase2_clinical_2ep_constant_1ep_decay_after_3ep_15mr/modernbert_phase2_clinical_2ep_constant_1ep_decay_after_3ep_15mr --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
# ./run_parallel.sh --dataset HOC --model /home/ubuntu/hf_models_new/phase2_clinical_3ep_decay_after_3ep_15mr/modernbert_phase2_clinical_3ep_decay_after_3ep_15mr --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
# ./run_parallel.sh --dataset HOC --model /home/ubuntu/hf_models_new/phase2_bio_1ep_decay_after_3ep_15mr/modernbert_phase2_bio_1ep_decay_after_3ep_15mr --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1

# ./run_parallel.sh --dataset HOC --model /home/ubuntu/hf_models_new/phase2_large_bio_1ep_decay_after_3ep_15mr/modernbert_phase2_large_bio_1ep_decay_after_3ep_15mr --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
# ./run_parallel.sh --dataset HOC --model /home/ubuntu/hf_models_new/phase2_large_clinical_2ep_constant_1ep_decay_after_3ep_15mr/modernbert_phase2_large_clinical_2ep_constant_1ep_decay_after_3ep_15mr --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
# ./run_parallel.sh --dataset HOC --model /home/ubuntu/hf_models_new/phase2_large_clinical_3ep_decay_after_3ep_15mr/modernbert_phase2_large_clinical_3ep_decay_after_3ep_15mr --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1

./run_parallel.sh --dataset HOC --model emilyalsentzer/Bio_ClinicalBERT --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
# ./run_parallel.sh --dataset HOC --model allenai/biomed_roberta_base --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
./run_parallel.sh --dataset HOC --model dmis-lab/biobert-v1.1 --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1

# ./run_parallel.sh --dataset HOC --model yikuan8/Clinical-Longformer --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
# ./run_parallel.sh --dataset HOC --model yikuan8/Clinical-BigBird --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1

# ./run_parallel.sh --dataset HOC --model Simonlee711/Clinical_ModernBERT --lr 7e-5 --wd 1e-5 --epochs 20 --batch_size 16 --accumulation_steps 1
