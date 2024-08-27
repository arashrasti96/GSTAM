gpu_id= # use 0-th gpu to run the experiments


dataset=NCI1; lr_adj=.01; lr_feat=0.01; outer=1;inner=0;lr_dist=0.1 ; attention_power=3; python main_attention.py  --attention_power ${attention_power} --lr_dist=${lr_dist} --outer=${outer} --inner=${inner} --dataset ${dataset} --init real  --gpu_id=${gpu_id} --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=10 --save=0


dataset=NCI1; lr_adj=1; lr_feat=0.005; outer=1;inner=0;lr_dist=0.1 ; attention_power=4; python main_attention.py  --attention_power ${attention_power} --outer=${outer} --inner=${inner} --dataset ${dataset} --init real  --gpu_id=${gpu_id} --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=10 --save=1


dataset=NCI1; lr_adj=1; lr_feat=0.005; outer=1;inner=0;lr_dist=0.1 ; attention_power=2; python main_attention.py  --attention_power ${attention_power} --outer=${outer} --inner=${inner} --dataset ${dataset} --init real  --gpu_id=1 --nconvs=3 --dis=mse --lr_adj=${lr_adj} --lr_feat=${lr_feat} --epochs=1000 --eval_init=1  --net_norm=none --pool=mean --seed=0 --ipc=50 --save=1