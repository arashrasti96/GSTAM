gpu_id=0



dataset=MUTAG; lr_adj=0.1; lr_feat=0.005; lr_dist=0.2 ; attention_power=2; python main_attention.py --attention_power ${attention_power} --lr_dist=${lr_dist} --dataset ${dataset} --init real  --gpu_id=${gpu_id} --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=1 --save=1


dataset=MUTAG; lr_adj=0.1; lr_feat=0.01; lr_dist=0.1 ; attention_power=3; python main_attention.py --attention_power ${attention_power} --lr_dist=${lr_dist} --dataset ${dataset} --init real  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=10 --save=1

dataset=MUTAG; lr_adj=0.2; lr_feat=0.005; lr_dist=0.1 ; attention_power=4; python main_attention.py --attention_power ${attention_power} --lr_dist=${lr_dist} --dataset ${dataset} --init real  --gpu_id=${gpu_id} --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=20 --save=0

