gpu=$1
data=$2

start=0
end=20
end=$((end-1))

for i in $(eval echo {$start..$end})
do
   python train_Kfold_CV.py --fold_id=$i --device $gpu --np_data_dir data\data-sleep-EDF-20-npz
done
#lr:1e-2