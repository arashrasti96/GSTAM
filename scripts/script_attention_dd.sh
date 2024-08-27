


dataset=DD; lr_adj=2; lr_feat=0.01; python main_attention.py --dataset ${dataset} --lr_dist 0.1 --attention_power 1 --beta 0.1 --init real --outer 1 --inner 0  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=1 --save=0


dataset=DD; lr_adj=2; lr_feat=0.05; python main_attention.py --dataset ${dataset} --lr_dist 0.1 --attention_power 1 --beta 0.1 --init real --outer 1 --inner 0  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=10 --save=0


dataset=DD; lr_adj=2; lr_feat=0.1; python main_attention.py --dataset ${dataset} --lr_dist 0.1 --attention_power 1 --beta 0.1 --init real --outer 1 --inner 0  --gpu_id=0 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=50 --save=0
