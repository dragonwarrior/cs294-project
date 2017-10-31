env=$1
if [[ -z $env ]];then
    env="Humanoid-v1"
fi
echo "gen data for env[$env]"
python3 ./run_expert.py ./experts/${env}.pkl "" --gen_data True --out_fname=./data/${env}
