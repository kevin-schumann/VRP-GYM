TSPModel=''
VRPModel=''
IRPModel=''
for data in TSP,$TSPMODEL VRP,$VRPModel IRP,$DEMANDModel do
    IFS=',' read  env model <<< "${data}"
    python reproduce.py --env_type $env \
            --model_path $model
done