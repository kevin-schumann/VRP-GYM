
#!/bin/sh

TSPModel='./check_points/tsp_20_69/model_epoch_850.pt'
VRPModel='./check_points/vrp_20_69/model_epoch_850.pt'
IRPModel='./check_points/irp_20_69/model_epoch_850.pt'
for data in TSP,$TSPModel VRP,$VRPModel IRP,$IRPModel; do
    IFS=',' read  env model <<< "${data}"
    python reproduction.py --env_type $env --model_path $model --num_nodes 20 --csv_path "./reproduction_log/reproduction_results_20_nodes_model_${env}.csv"
done

TSPModel='./check_points/tsp_30_69/model_epoch_850.pt'
VRPModel='./check_points/vrp_30_69/model_epoch_850.pt'
IRPModel='./check_points/irp_30_69/model_epoch_850.pt'
for data in TSP,$TSPModel VRP,$VRPModel IRP,$IRPModel; do
    IFS=',' read  env model <<< "${data}"
    python reproduction.py --env_type $env --model_path $model --num_nodes 30 --csv_path "./reproduction_log/reproduction_results_30_nodes_model_${env}.csv"
done

TSPModel='./check_points/tsp_40_69/model_epoch_850.pt'
VRPModel='./check_points/vrp_40_69/model_epoch_850.pt'
IRPModel='./check_points/irp_40_69/model_epoch_850.pt'
for data in TSP,$TSPModel VRP,$VRPModel IRP,$IRPModel; do
    IFS=',' read  env model <<< "${data}"
    python reproduction.py --env_type $env --model_path $model --num_nodes 40 --csv_path "./reproduction_log/reproduction_results_40_nodes_model_${env}.csv"
done

# try to apply model trained on 20 nodes in 40 nodes
TSPModel='./check_points/tsp_20_69/model_epoch_850.pt'
VRPModel='./check_points/vrp_20_69/model_epoch_850.pt'
IRPModel='./check_points/irp_20_69/model_epoch_850.pt'
for data in TSP,$TSPModel VRP,$VRPModel IRP,$IRPModel; do
    IFS=',' read  env model <<< "${data}"
    python reproduction.py --env_type $env --model_path $model --num_nodes 40 --csv_path "./reproduction_log/reproduction_20_in_40_nodes_model_${env}.csv"
    
done