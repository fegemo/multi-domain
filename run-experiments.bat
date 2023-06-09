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

::echo "Starting ALL"
::python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30 --no-aug  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-off,tran-off
::python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30 --no-hue  --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-off,tran-on
::python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30 --no-tran --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-on,tran-off
::python train.py baseline --rm2k --rmxp --rmvx --tiny --misc --lambda_l1 30           --callback-evaluate-fid --callback-evaluate-l1 --batch 4 --epochs 240 --log-folder temp-side2side/augmentation/all,hue-on,tran-on
::echo "Finished ALL"

::echo "Starting LAMBDA_PALETTE STUDY"
::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss100rise --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1
::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss200rise --lambda-palette 200 --callback-evaluate-fid --callback-evaluate-l1
::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss0       --lambda-palette 0   --callback-evaluate-fid --callback-evaluate-l1
::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss10rise  --lambda-palette 10  --callback-evaluate-fid --callback-evaluate-l1
::echo "Finished LAMBDA_PALETTE STUDY"


::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss100rise-tv1em3 --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --lambda-tv 0.001
::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss100rise-tv1em2 --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --lambda-tv 0.01
::python train.py stargan-paired --rm2k --log-folder output --epochs 300 --no-aug --d-steps 1 --model-name stargan-paired-palette --experiment paletteloss100rise-tv5em2 --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --lambda-tv 0.05
python train.py stargan-paired --rm2k --log-folder output --epochs 350 --no-tran --d-steps 1 --model-name stargan-paired-paletteandtv --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --source-domain-aware-generator --conditional-discriminator --sampler multi-target --experiment paletteloss100rise --lambda-tv 0
python train.py stargan-paired --rm2k --log-folder output --epochs 350 --no-tran --d-steps 1 --model-name stargan-paired-paletteandtv --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --source-domain-aware-generator --conditional-discriminator --sampler multi-target --experiment paletteloss100rise-tv2em3rise --lambda-tv 0.001
python train.py stargan-paired --rm2k --log-folder output --epochs 350 --no-tran --d-steps 1 --model-name stargan-paired-paletteandtv --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --source-domain-aware-generator --conditional-discriminator --sampler multi-target --experiment paletteloss100rise-tv2em3rise --lambda-tv 0.002
python train.py stargan-paired --rm2k --log-folder output --epochs 350 --no-tran --d-steps 1 --model-name stargan-paired-paletteandtv --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --source-domain-aware-generator --conditional-discriminator --sampler multi-target --experiment paletteloss100rise-tv3em3rise --lambda-tv 0.003
python train.py stargan-paired --rm2k --log-folder output --epochs 350 --no-tran --d-steps 1 --model-name stargan-paired-paletteandtv --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --source-domain-aware-generator --conditional-discriminator --sampler multi-target --experiment paletteloss100rise-tv8em4rise --lambda-tv 0.0008
python train.py stargan-paired --rm2k --log-folder output --epochs 350 --no-tran --d-steps 1 --model-name stargan-paired-paletteandtv --lambda-palette 100 --callback-evaluate-fid --callback-evaluate-l1 --source-domain-aware-generator --conditional-discriminator --sampler multi-target --experiment paletteloss100rise-tv5em4rise --lambda-tv 0.0005
