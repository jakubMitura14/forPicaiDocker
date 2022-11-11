echo "staaarting "
# docker run --init --gpus all --ipc host --privileged --net host -p 8888:8888 -p49053:49053 --restart unless-stopped -v /mnt/disks/sde:/home/sliceruser/data -v /mnt/disks/sdb:/home/sliceruser/dataOld -it  slicerpicai:latest bash

cd /home/sliceruser/locTemp/picai_baseline
git config --global --add safe.directory /home/sliceruser/locTemp/picai_baseline
git pull
# git switch LnMonoB
git switch ${branch}
git pull
# python3.9 -u /home/sliceruser/locTemp/picai_baseline/src/picai_baseline/unet/train.py \
#   --weights_dir='/home/sliceruser/locTemp/workdirSemiOpi/results/UNet/weights/' \
#   --overviews_dir='/home/sliceruser/locTemp/workdirSemiOpi/results/UNet/overviews/' \
#   --folds 0 1 2 3 4 --max_threads 12 --enable_da 1 --num_epochs 250 --batch_size 48 \
#   --validate_n_epochs 1 --validate_min_epoch 0

python3.9 -u /home/sliceruser/locTemp/picai_baseline/src/picai_baseline/unet/train.py \
  --weights_dir='/home/sliceruser/workdir/results/UNet/weights/' \
  --overviews_dir='/home/sliceruser/workdir/results/UNet/overviews/' \
  --folds 0 1 2 3 4 --max_threads 12 --enable_da 1 --num_epochs 250 --batch_size 48 \
  --validate_n_epochs 1 --validate_min_epoch 0




# git config --global --add safe.directory /home/sliceruser/locTemp/piCaiCode
# cd /home/sliceruser/locTemp/piCaiCode
# git pull
# python3.8 /home/sliceruser/locTemp/piCaiCode/Three_chan_baseline_hyperParam.py
#python3.8 /home/sliceruser/data/piCaiCode/exploration/rayPlay/rayExampleBasic.py


