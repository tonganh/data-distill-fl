# Non-iid experiments
# Clientdirect
# python main_mobile.py --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --algorithm fed_client_direct --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --weight_decay 0.001 --proportion 1 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 1
# python main_mobile.py --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 1 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 3
python main_mobile.py --edge_update_frequency 3 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 6
python main_mobile.py --edge_update_frequency 6 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 9
# python main_mobile.py --edge_update_frequency 9 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 50 --std_velocity 10  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 6, velocity = 0 (static)
python main_mobile.py --edge_update_frequency 6 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 0 --std_velocity 0  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 6, random coordinates
# python main_mobile.py --edge_update_frequency 6 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 0 --std_velocity 0  --sample active --model cnn --algorithm rand_fed_edgeavg --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 6, velocity = 100
python main_mobile.py --edge_update_frequency 6 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 100 --std_velocity 10  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 6, velocity = 200
# python main_mobile.py --edge_update_frequency 6 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 200 --std_velocity 10  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
# EdgeAvg, frequency = 6, velocity = 10
python main_mobile.py --edge_update_frequency 6 --task  cifar10_cnum60_dist2_skew0.05_seed0 --non_iid_classes 1 --mean_velocity 10 --std_velocity 2  --sample active --model cnn --algorithm fed_mobile_distill --num_edges 5 --num_clients 50 --std_num_clients 10 --num_rounds 300 --num_epochs 3 --learning_rate 0.001 --momentum 0.95 --proportion 0.5 --batch_size 64 --eval_interval 1 --gpu 0 --server_gpu_id 0 --num_threads 1
