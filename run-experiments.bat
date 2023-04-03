@echo off
::echo "Starting RM2K"
::python train.py baseline --rm2k --lambda_l1 30 --no-aug  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rm2k,hue-off,tran-off
::python train.py baseline --rm2k --lambda_l1 30 --no-hue  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rm2k,hue-off,tran-on
::python train.py baseline --rm2k --lambda_l1 30 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rm2k,hue-on,tran-off
::python train.py baseline --rm2k --lambda_l1 30           --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rm2k,hue-on,tran-on
::echo "Finished RM2K"

::echo "Starting RMXP"
::python train.py baseline --rmxp --lambda_l1 30 --no-aug  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmxp,hue-off,tran-off
::python train.py baseline --rmxp --lambda_l1 30 --no-hue  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmxp,hue-off,tran-on
::python train.py baseline --rmxp --lambda_l1 30 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmxp,hue-on,tran-off
::python train.py baseline --rmxp --lambda_l1 30           --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmxp,hue-on,tran-on
::echo "Finished RMXP"

::echo "Starting RMVX"
::python train.py baseline --rmvx --lambda_l1 30 --no-aug  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmvx,hue-off,tran-off
::python train.py baseline --rmvx --lambda_l1 30 --no-hue  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmvx,hue-off,tran-on
::python train.py baseline --rmvx --lambda_l1 30 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmvx,hue-on,tran-off
::python train.py baseline --rmvx --lambda_l1 30           --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/rmvx,hue-on,tran-on
::echo "Finished RMVX"

::echo "Starting TINY"
::python train.py baseline --tiny --lambda_l1 30 --no-aug  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/tiny,hue-off,tran-off
::python train.py baseline --tiny --lambda_l1 30 --no-hue  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/tiny,hue-off,tran-on
::python train.py baseline --tiny --lambda_l1 30 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/tiny,hue-on,tran-off
::python train.py baseline --tiny --lambda_l1 30           --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/tiny,hue-on,tran-on
::echo "Finished TINY"

echo "Starting ALL"
python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30 --no-aug  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-off,tran-off
python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30 --no-hue  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-off,tran-on
python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-on,tran-off
python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30           --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-on,tran-on
echo "Finished ALL"