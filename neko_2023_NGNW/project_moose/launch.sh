# sh launch.sh [exp_folder] [gpuid]
LOG_ROOT=/home/lasercat/hydra_logs/
CFG=${PWD}/"gpu_cat.json"

EXP=${1}
GID=${2}

#IDRUN=$(uuidgen)
#echo "Create dir for log"
CURRENTDATE=$(date +"%F-%H-%M-%S")
TAG=${EXP}_${GID}_${CURRENTDATE}
echo "currentDate :"
echo $CURRENTDATE
PATHLOGDIR="${LOG_ROOT}/${TAG}"
echo "path log dir:"
echo ${PATHLOGDIR}
mkdir ${PATHLOGDIR}
PATHLOG="${PATHLOGDIR}/PLAYDAN.log"

export PYTHONPATH=../../../
#">&1 means to redirect stderr(2) to stdout(1)

cd ${EXP};
export CUDA_VISIBLE_DEVICES=${GID};
screen -dmS ${TAG} python3 train_osr_hv_cat_wandb.py ${CFG} 2>&1 | tee ${PATHLOG}
