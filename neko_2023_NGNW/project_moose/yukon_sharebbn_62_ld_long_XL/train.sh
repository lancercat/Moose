#Cmd to launch
#train.sh [gpuid]
conda activate myenv

IDRUN=uuidgen
echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOGDIR="home/simcor/dev/logs/mose/${CURRENTDATE}_ID_${IDRUN}/"
echo "path log dir:"
echo PATHLOGDIR
mkdir PATHLOGDIR

PATHLOG="${PATHLOGDIR}PLAYDAN.log"

export PYTHONPATH=../../../

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$1

PATHCONFIGGPU="/home/simcor/dev/projects/mose/neko_2023_NGNW/project_moose_dgx/hydra_62_ld_long/gpu_simon.json"

#">&1 means to redirect stderr(2) to stdout(1)
python3 train_osr_hv.py PATHCONFIGGPU 2>&1 | tee PATHLOG



