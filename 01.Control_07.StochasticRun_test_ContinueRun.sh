#!/bin/bash
#SBATCH -C "gpu&hbm80g" 
#SBATCH -A mxxx_g
#SBATCH -q regular
#SBATCH --time=14:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind none
#SBATCH --module=gpu,nccl-plugin
#SBATCH -J 02.StochasticRun_test_ContinueRun
#SBATCH -o ./logs/02.Control_07_${iyr}${i_Mon}${iDD}_%j.out

## Author: Ziming Chen at PNNL, Ya Wang & Letian Gu at IAP CAS; Date: 2026-05-22
## This code control python script: 02.StochasticRun_test_ContinueRun.py

start_time=$(date +%s)
echo "Start Time : $(date -d @${start_time} +'%Y-%m-%d %H:%M:%S')"

conda activate py_ai2

iyr=2025
i_Mon=12
iDD=01
echo "Date: "${iyr}"-"${i_Mon}"-"${iDD}

SimulationPeriod=( "2025-12-01" "2025-12-10" )
Sim_Year=( $(echo ${SimulationPeriod[0]} | cut -d'-' -f1) $(echo ${SimulationPeriod[1]} | cut -d'-' -f1) )
Sim_Mon=( $(echo ${SimulationPeriod[0]} | cut -d'-' -f2) $(echo ${SimulationPeriod[1]} | cut -d'-' -f2) )
Sim_DD=( $(echo ${SimulationPeriod[0]} | cut -d'-' -f3) $(echo ${SimulationPeriod[1]} | cut -d'-' -f3) )
echo "Simulation Period: "${Sim_Year[0]}"-"${Sim_Mon[0]}"-"${Sim_DD[0]}" to "${Sim_Year[1]}"-"${Sim_Mon[1]}"-"${Sim_DD[1]}

python ./02.Stochastic_ContinuousRun_IndicateInitCon.py \
  --s_yr_Init ${iyr} --s_mon_Init ${i_Mon} --s_DD_Init ${iDD} \
  --Sim_Year_Begin ${Sim_Year[0]} --Sim_Mon_Begin ${Sim_Mon[0]} --Sim_DD_Begin ${Sim_DD[0]} \
  --Sim_Year_End ${Sim_Year[1]} --Sim_Mon_End ${Sim_Mon[1]} --Sim_DD_End ${Sim_DD[1]} \
  2>&1 | tee ./logs/02.ContinueRun_${iyr}${i_Mon}${iDD}_${i_UnifWarming}K_%j.log

# Uniform warming run
# i_UnifWarming=1
# python ./02.Stochastic_ContinuousRun_IndicateInitCon.py \
#   --s_yr_Init ${iyr} --s_mon_Init ${i_Mon} --s_DD_Init ${iDD} --i_UnifWarming ${i_UnifWarming} \
#   --Sim_Year_Begin ${Sim_Year[0]} --Sim_Mon_Begin ${Sim_Mon[0]} --Sim_DD_Begin ${Sim_DD[0]} \
#   --Sim_Year_End ${Sim_Year[1]} --Sim_Mon_End ${Sim_Mon[1]} --Sim_DD_End ${Sim_DD[1]} \
#   2>&1 | tee ./logs/02.ContinueRun_${iyr}${i_Mon}${iDD}_${i_UnifWarming}K_%j.log

end_time=$(date +%s)
echo "End Time : $(date -d @${end_time} +'%Y-%m-%d %H:%M:%S')"
echo "Total Time : $((end_time - start_time)) seconds"