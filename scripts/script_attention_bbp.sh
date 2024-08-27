


dataset=ogbg-molbbbp; lr_adj=1; lr_feat=0.001; python main_attention.py --dataset ${dataset} --lr_dist 0.1 --attention_power 4 --init real  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=1 --save=0


dataset=ogbg-molbbbp; lr_adj=1; lr_feat=0.003; python main_attention.py --dataset ${dataset} --lr_dist 0.1 --attention_power 4 --init real  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=10 --save=0


dataset=ogbg-molbbbp; lr_adj=1; lr_feat=0.001; python main_attention.py --dataset ${dataset} --lr_dist 0.1 --attention_power 4 --init real  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --i