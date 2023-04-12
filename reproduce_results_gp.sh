

#for pb in CartPole Acrobot MountainCar Pendulum BipedalWalker BipedalWalkerHardcore 
#do
#python -c "import pybullet ; import gym ; import pybullet ; gym.make('$pb-v2')"  2>&1 | tail -n 1 | sed 's/...$//g'  | sed "s/.*'//g"
#
#done
#echo HopperBulletEnv-v0
#echo InvertedDoublePendulum-v2
#echo 'Not found: InvPendulumSwingUp ? Not pendulum ?'
#echo LunarLander-v2

stamp=STAMP${RANDOM}_${RANDOM}_`date | sed 's/ /_/g'`
for pb in `cat listpb.txt`
do
(
(
    if [[ "$pb" == *"Mountain"* ]]; then
      conf="conf/conf_gp_3.yml"
    elif [[ "$pb" == *"CartPole"* ]] || [[ "$pb" == *"Acrobot"* ]] || [[ "$pb" == "Pendulum-v0" ]]; then
         conf="conf/conf_gp_124.yml"
    elif [[ "$pb" == *""* ]]; then
         conf="conf/conf_gp_5678910.yml"
    fi
    filename="conf/conf_gp_${pb}_${stamp}.yml"
    cp $conf $filename
    sed -i "s/env:.*/env: $pb/g" $filename
    python evolve.py --conf $filename &
    if [[ "$pb" == *"Mountain"* ]]; then
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    python evolve.py --conf $filename &
    fi
    wait
) | tee run_$stamp
) &
done

wait

#conf_gp.yml     conf_gpUCB_124.yml  conf_gpUCB_5678910.yml  conf_qdgp-BipedalWalker.yml  conf_qdlingp-BipedalWalker.yml
#conf_gpUCB.yml  conf_gpUCB_3.yml    conf_lingp.yml          conf_qdgp-Hopper.yml         conf_qdlingp-Hopper.yml
