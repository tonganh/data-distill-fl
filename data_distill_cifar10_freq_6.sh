# !10 mean 1/20 tập dữ liệu
python main_distill.py --edge_update_frequency 6 --algorithm fed_distill_kip --kip_support_size 10 \
 --distill_iters 3000 --distill_data_path "distill_data_kip/"  --model resnet9_custom --weight_decay 1e-3 \
 --task cifar10_cnum100_dist8_skew0.8_seed0 --mean_velocity 200 --std_velocity 10  --sample uniform --num_edges 5 \
 --num_clients 100 --std_num_clients 10 --num_rounds 200 --num_epochs 5 --learning_rate 0.01 \
 --momentum 0.9 --proportion 0.3 \
 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1 --learning_rate_decay 0.9 --distill_ipc 10 --dropout_value=0.3
# python main_distill.py --edge_update_frequency 6 --algorithm fed_distill_kip --distill_before_train 1 --kip_support_size 50 --distill_iters 500 --distill_data_path "distill_data_kip/"  --model resnet9 --weight_decay 1e-4 --task cifar10_cnum50_dist0_skew0_seed0 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --num_edges 10 --num_clients 100 --std_num_clients 10 --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --momentum 0.9 --proportion 0.3 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 1 --num_threads 1
# python main_distill.py --edge_update_frequency 1 --algorithm fed_distill_kip --distill_before_train 1 --kip_support_size 50 --distill_iters 500 --distill_data_path "distill_data_kip/"  --model resnet9 --weight_decay 1e-4 --task cifar10_cnum50_dist0_skew0_seed0 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --num_edges 10 --num_clients 100 --std_num_clients 10 --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --momentum 0.9 --proportion 0.3 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 1 --num_threads 1
