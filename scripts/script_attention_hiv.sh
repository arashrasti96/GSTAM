

dataset=ogbg-molhiv; lr_adj=0.01; lr_feat=0.01; python main_attention.py --dataset ${dataset} --init real  --lr_dist 0.1 --attention_power 2 --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=sum --seed=0 --ipc=1 --save=0


dataset=ogbg-molhiv; lr_adj=0.01; lr_feat=0.01; python main_attention.py --dataset ${dataset} --init real  --lr_dist 0.1 --attention_power 4 --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=sum --seed=0 --ipc=10 --save=0


dataset=ogbg-molhiv; lr_adj=0.01; lr_feat=0.01; python main_attention.py --dataset ${dataset} --init real  --lr_dist 0.1 --attention_power 1 --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=sum --seed=0 --ipc=50 --save=0 