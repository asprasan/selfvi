# result_dir="../CleanCode/results/for_git/"
# expt_dir="no_stereo_uv77-unet_icip-Aug-16-14:26:46"
# save_dir="$result_dir$expt_dir"
python -W ignore test_lf.py \
--rank 12 \
--seq-len 4 \
--layers 3 \
--angular 7 \
--display multilayer \
--model unet_icip \
--h5-file insert h5 file path here \
--dataset test_lf \
--disp-model dispnetC \
--inph 352 \
--inpw 512 \
--psnr \
--flow-model flownetC \
--gpu 2 \
--batch_size 1 \
--dry-run \
--test \
--resume-training \
--restore-file "data/checkpoint.pt" \
--save-dir "results"