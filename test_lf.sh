result_dir="../CleanCode/results/for_git/"
expt_dir="no_stereo_uv77-unet_icip-Aug-16-14:26:46"
save_dir="$result_dir$expt_dir"
python -W ignore test_lf.py \
--rank 12 \
--seq-len 4 \
--layers 3 \
--angular 7 \
--output-dir /data/prasan/hdr/CleanCode/results \
--display multilayer \
--model unet_icip \
--h5-file hybrid/hybrid_test.h5 \
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
--restore-file "weights/checkpoint_last.pt" \
--save-dir "results"
#--restore-file "$save_dir/checkpoints/checkpoint_last.pt" \
# python lf2video.py $save_dir
# Raytrix/Raytrix_test
# hybrid/hybrid_test
# kalantari/kalan_LFvid_test
# train/lfvideo_train