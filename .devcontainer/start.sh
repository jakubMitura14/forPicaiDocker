echo "staaarting "
# docker run --init --gpus all --ipc host --privileged --net host -p 8888:8888 -p49053:49053 --restart unless-stopped -v /mnt/disks/sde:/home/sliceruser/data -v /mnt/disks/sdb:/home/sliceruser/dataOld -it  slicerpicai:latest bash
cd /home/sliceruser/data/piCaiCode
git pull
python3.8 /home/sliceruser/locTemp/piCaiCode/Three_chan_baseline_hyperParam.py
#python3.8 /home/sliceruser/data/piCaiCode/exploration/rayPlay/rayExampleBasic.py


