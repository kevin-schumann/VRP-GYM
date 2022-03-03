TSPModel=''
VRPModel=''
for data in TSP,$TSPMODEL VRP,$VRPModel DEMANDVRP,$DEMANDModel do
    IFS=',' read  env model <<< "${data}"
    python reproduce.py --env_type $env \
            --net $model
done