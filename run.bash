lrs=($(seq 0.07 0.033 0.4))
lambs=($(seq 1 5))
gammas=($(seq 1 5))

for lr in "${lrs[@]}"; do 
    for lam in "${lambs[@]}"; do
        for gamma in "${gammas[@]}"; do 
            log_name=fair_ssl_hsic_lr${lr}_lam${lam}_gamma${gamma}
            python main.py --model fair-ssl-hsic --lr ${lr} --lamb ${lam} --gamma ${gamma} --wandb_name ${log_name}
        done
    done
done