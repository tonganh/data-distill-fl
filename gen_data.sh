# python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
# python generate_fedtask.py --dataset mnist --dist 1 --skew 0.2 --num_clients 60
# python generate_fedtask.py --dataset cifar10 --dist 1 --skew 0.2 --num_clients 60
python generate_fedtask.py --dataset cifar10 --dist 2 --skew 0.05 --num_clients 60
python generate_fedtask.py --dataset mnist --dist 2 --skew 0.05 --num_clients 60
python generate_fedtask.py --dataset cifar100 --dist 2 --skew 0.05 --num_clients 60

