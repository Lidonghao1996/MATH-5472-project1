CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 25.0 --gamma 1000 --manualSeed 1 --save_dir dsprites_image &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 32 --C_max 50.0 --gamma 1000 --manualSeed 1  --save_dir celeba_image &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss normal --epoch 74 --C_max 25.0 --gamma 1000 --manualSeed 1 --save_dir 3dchairs_image &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 25.0 --gamma 1000 --manualSeed 2 --save_dir dsprites_image2 &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 32 --C_max 50.0 --gamma 1000 --manualSeed 2  --save_dir celeba_image2 &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss normal --epoch 74 --C_max 25.0 --gamma 1000 --manualSeed 2 --save_dir 3dchairs_image2 &

CUDA_VISIBLE_DEVICES=1 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 25.0 --gamma 100 --manualSeed 2 --save_dir dsprites_image2_gamma100 &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --loss normal --epoch 32 --C_max 50.0 --gamma 100 --manualSeed 2  --save_dir celeba_image2_gamma100 &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset 3dchairs --loss normal --epoch 74 --C_max 25.0 --gamma 100 --manualSeed 2 --save_dir 3dchairs_image2_gamma100 &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 25.0 --gamma 100 --manualSeed 2 --save_dir dsprites_image2_gamma100_nats &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 32 --C_max 50.0 --gamma 100 --manualSeed 2  --save_dir celeba_image2_gamma100_nats &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss normal --epoch 74 --C_max 25.0 --gamma 100 --manualSeed 2 --save_dir 3dchairs_image2_gamma100_nats &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 25.0 --gamma 1 --manualSeed 2 --save_dir dsprites_image2_gamma1_nats &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 32 --C_max 50.0 --gamma 1 --manualSeed 2  --save_dir celeba_image2_gamma1_nats &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss normal --epoch 74 --C_max 25.0 --gamma 1 --manualSeed 2 --save_dir 3dchairs_image2_gamma1_nats &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 3.21 --gamma 1 --manualSeed 2 --save_dir dsprites_image2_gamma1_nats &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 32 --C_max 3.91 --gamma 1 --manualSeed 2  --save_dir celeba_image2_gamma1_nats &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 74 --C_max 3.21 --gamma 1 --manualSeed 2 --save_dir 3dchairs_image2_gamma1_nats &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 3.21 --gamma 1 --manualSeed 2 --save_dir dsprites_image2_gamma1_bernuolli &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 32 --C_max 3.91 --gamma 1 --manualSeed 2  --save_dir celeba_image2_gamma1_bernuolli &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 74 --C_max 3.21 --gamma 1 --manualSeed 2 --save_dir 3dchairs_image2_gamma1_bernuolli &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 3.21 --gamma 100 --manualSeed 2 --save_dir dsprites_image2_gamma100_bernuolli &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 32 --C_max 3.91 --gamma 100 --manualSeed 2  --save_dir celeba_image2_gamma100_bernuolli &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 74 --C_max 3.21 --gamma 100 --manualSeed 2 --save_dir 3dchairs_image2_gamma100_bernuolli &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir dsprites_image2_gamma10_bernuolli &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 32 --C_max 3.91 --gamma 10 --manualSeed 3  --save_dir celeba_image2_gamma10_bernuolli &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 74 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir 3dchairs_image2_gamma10_bernuolli &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 10 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir dsprites_image2_gamma10_bernuolli_more &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 32 --C_max 3.91 --gamma 10 --manualSeed 3  --save_dir celeba_image2_gamma10_bernuolli_more &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 74 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir 3dchairs_image2_gamma10_bernuolli_more &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir dsprites_gamma10_bernuolli_more_epoch &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 10 --manualSeed 3  --save_dir celeba_gamma10_bernuolli_more_epoch &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir 3dchairs_gamma10_bernuolli_more_epoch &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 25 --gamma 10 --manualSeed 3 --save_dir dsprites_gamma10_c25_bernuolli_more_epoch &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 50 --gamma 10 --manualSeed 3  --save_dir celeba_gamma10_c50_bernuolli_more_epoch &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 25 --gamma 10 --manualSeed 3 --save_dir 3dchairs_gamma10_c25_bernuolli_more_epoch &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir dsprites_gamma10_bernuolli_more_epoch_image &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 10 --manualSeed 3  --save_dir celeba_gamma10_bernuolli_more_epoch_image &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir 3dchairs_gamma10_bernuolli_more_epoch_image &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir dsprites_gamma10_bernuolli_resize &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 10 --manualSeed 3  --save_dir celeba_gamma10_bernuolli_resize &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir 3dchairs_gamma10_bernuolli_resize &

C max iteration
CUDA_VISIBLE_DEVICES=1 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir dsprites_gamma10_bernuolli_resize_citer &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 10 --manualSeed 3  --save_dir celeba_gamma10_bernuolli_resize_citer &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 10 --manualSeed 3 --save_dir 3dchairs_gamma10_bernuolli_resize_citer &

hinge loss 
CUDA_VISIBLE_DEVICES=1 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 10 --manualSeed 3 --hinge_loss 1 --save_dir dsprites_gamma10_bernuolli_resize_citer_hinge_loss &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 10 --manualSeed 3  --hinge_loss 1 --save_dir celeba_gamma10_bernuolli_resize_citer_hinge_loss &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 10 --manualSeed 3 --hinge_loss 1 --save_dir 3dchairs_gamma10_bernuolli_resize_citer_hinge_loss &

hinge loss 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 10 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir dsprites_gamma10_bernuolli_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 10 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma10_bernuolli_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 10 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir 3dchairs_gamma10_bernuolli_resize_citer100000_hinge_loss &

hinge loss 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir dsprites_gamma100_bernuolli_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 100 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma100_bernuolli_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir 3dchairs_gamma100_bernuolli_resize_citer100000_hinge_loss &

hinge loss 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 3.21 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir dsprites_gamma1000_bernuolli_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 3.91 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma1000_bernuolli_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 50 --C_max 3.91 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma1000_normal_resize_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 3.21 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir 3dchairs_gamma1000_bernuolli_resize_citer100000_hinge_loss &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir dsprites_gamma1000_bernuolli_c25_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma1000_bernuolli_c50_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 50 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma1000_normal_c50_citer100000_hinge_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir 3dchairs_gamma1000_bernuolli_c25_citer100000_hinge_loss &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir dsprites_gamma1000_bernuolli_c25_citer100000_abs_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 0 --save_dir celeba_gamma1000_bernuolli_c50_citer100000_abs_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss normal --epoch 50 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 0 --save_dir celeba_gamma1000_normal_c50_citer100000_abs_loss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir 3dchairs_gamma1000_bernuolli_c25_citer100000_abs_loss &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 15 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir dsprites_gamma1000_bernuolli_c25_citer100000_abs_loss_batch_size256 &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --loss bernoulli --epoch 50 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 0 --save_dir celeba_gamma1000_bernuolli_c50_citer100000_abs_loss_batch_size256 &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --loss normal --epoch 50 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 0 --save_dir celeba_gamma1000_normal_c50_citer100000_abs_loss_batch_size256 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset 3dchairs --loss bernoulli --epoch 120 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir 3dchairs_gamma1000_bernuolli_c25_citer100000_abs_loss_batch_size256 &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir dsprites_gamma100_bernuolli_c25_citer100000_abs_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset celeba --loss bernoulli --epoch 200 --C_max 50 --gamma 10 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma10_bernuolli_c50_citer100000_hinge_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset celeba --loss normal --epoch 200 --C_max 50 --gamma 10 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 1 --save_dir celeba_gamma10_normal_c50_citer100000_hinge_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset 3dchairs --loss bernoulli --epoch 480 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir 3dchairs_gamma10_bernuolli_c25_citer100000_hinge_loss_batchsize256 &

CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir dsprites_gamma100_bernuolli_c25_citer100000_abs_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 20 --gamma 10 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir dsprites_gamma10_bernuolli_c10_citer100000_abs_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 20 --gamma 10 --manualSeed 3 --C_stop_iter 10000 --hinge_loss 0 --save_dir dsprites_gamma10_bernuolli_c10_citer10000_abs_loss_batchsize256 &

CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir dsprites_gamma10_bernuolli_c25_citer100000_abs_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10 --manualSeed 3 --C_stop_iter 10000  --hinge_loss 0 --save_dir dsprites_gamma10_bernuolli_c25_citer10000_abs_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir dsprites_gamma10_bernuolli_c25_citer100000_hinge_loss_batchsize256 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10 --manualSeed 3 --C_stop_iter 10000  --hinge_loss 1 --save_dir dsprites_gamma10_bernuolli_c25_citer10000_hinge_loss_batchsize256 &

#希望c step可以解决logvar向负无穷增长的情况 是可以的，但是问题在于重构图像还是不清楚
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1000 --save_dir ds_gamma100_ber_c25_citer100000_abs_loss_c_step_1000 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1    --save_dir ds_gamma100_ber_c25_citer100000_abs_loss_c_step_1    &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1000 --save_dir ds_gamma10_ber_c25_citer100000_abs_loss_c_step_1000 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1    --save_dir ds_gamma10_ber_c25_citer100000_abs_loss_c_step_1    &

# 加入了test set 并且解决了c不存在的问题
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1000 --save_dir Fix_ds_gamma100_ber_c25_citer100000_abs_loss_c_step_1000 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1    --save_dir Fix_ds_gamma100_ber_c25_citer100000_abs_loss_c_step_1    &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1000 --save_dir Fix_ds_gamma10_ber_c25_citer100000_abs_loss_c_step_1000 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 1    --save_dir Fix_ds_gamma10_ber_c25_citer100000_abs_loss_c_step_1    &

# 为什么提前达到最大的c有好处？ 是不是可以增加C——step？
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 10000 --save_dir Fix_ds_gamma100_ber_c25_citer100000_abs_loss_c_step_10000 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 2000    --save_dir Fix_ds_gamma100_ber_c25_citer100000_abs_loss_c_step_2000    &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 10000 --save_dir Fix_ds_gamma10_ber_c25_citer100000_abs_loss_c_step_10000 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 2000 --save_dir Fix_ds_gamma10_ber_c25_citer100000_abs_loss_c_step_2000    &

# 试一试不约束kl gamma=0 这个时候需要overfitting数据
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 0 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --C_step 10000 --save_dir Fix_ds_gamma0_ber_0 &
# 可以优化的，为什么后面不行了呢？因为前期太死了吗？
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 20000 --hinge_loss 0 --C_step 1000 --C_start 1.0 --save_dir Fix_ds_gamma100_ber_c1_25_citer20000_abs_loss_c_step_1000 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 100 --manualSeed 3 --C_stop_iter 20000 --hinge_loss 0 --C_step 1    --C_start 1.0 --save_dir Fix_ds_gamma100_ber_c1_25_citer20000_abs_loss_c_step_1    &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 20000 --hinge_loss 0 --C_step 1000 --C_start 1.0 --save_dir Fix_ds_gamma10_ber_c1_25_citer20000_abs_loss_c_step_1000 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10  --manualSeed 3 --C_stop_iter 20000 --hinge_loss 0 --C_step 1    --C_start 1.0 --save_dir Fix_ds_gamma10_ber_c1_25_citer20000_abs_loss_c_step_1    &

# 随机种子有问题

# Adam 是不是有问题？
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 10 --manualSeed 3 --C_stop_iter 20000 --hinge_loss 0 --C_step 1000 --C_start 1.0 --save_dir Fix_ds_gamma10_ber_c1_25_citer20000_abs_loss_c_step_1000_adam &

CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 0 --gamma 1 --manualSeed 3 --C_start 0 --save_dir Fix_ds_VAE &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 0 --gamma 10 --manualSeed 3 --C_start 0 --save_dir Fix_ds_BetaVAE &
# the loss functions would be sum!
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 0 --gamma 1 --manualSeed 3 --C_start 0 --save_dir Sum_ds_VAE &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 0 --gamma 10 --manualSeed 3 --C_start 0 --save_dir Sum_ds_BetaVAE &

# 解决了klmean dewenti
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 0 --gamma 1 --manualSeed 3 --C_start 0 --save_dir Sum_ds_VAE_mean1 &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 0 --gamma 10 --manualSeed 3 --C_start 0 --save_dir Sum_ds_BetaVAE_mean1 &

# 尝试各种setting！(defaught的cstep是1000)
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c25 &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset celeba --loss bernoulli --epoch 200 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 0 --save_dir New_celeba_gamma1000_ber_c50 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset celeba --loss normal --epoch 200 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 100000 --hinge_loss 0 --save_dir New_celeba_gamma1000_normal_c50 &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 480 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 0 --save_dir New_3dchairs_gamma1000_bernuolli_c25 &

# 我之前想了一些什么呢？Hinge loss？
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir New_ds_gamma1000_ber_c25_hinge_loss_new &
CUDA_VISIBLE_DEVICES=3 python train.py --dataset 3dchairs --loss bernoulli --epoch 480 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir New_3dchairs_gamma1000_bernuolli_c25_hinge_loss_new &
# fix outof bound
CUDA_VISIBLE_DEVICES=3 python train.py --dataset 3dchairs --loss bernoulli --epoch 480 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 100000 --hinge_loss 1 --save_dir New_3dchairs_gamma1000_bernuolli_c25_hinge_loss_new &

## cstep有什么作用？

# 剩下两个数据集可能需要调整 celeba已经可以了
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c25_cstop30k &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 480 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_3dchairs_gamma1000_bernuolli_c25_cstop30k &

# 怎么调节随机种子 ds数据集？
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c25_cstop30k_random &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset 3dchairs --loss bernoulli --epoch 480 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 1 --save_dir New_3dchairs_gamma1000_bernuolli_c25_cstop30k_hingeloss &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 1 --save_dir New_ds_gamma1000_ber_c25_cstop30k_random_hingeloss &


# 剩下的用人脸可以展示path！
CUDA_VISIBLE_DEVICES=2 python train.py --dataset celeba --loss normal --epoch 400 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 300000 --hinge_loss 0 --save_dir New_celeba_gamma1000_normal_c50_cstep30k_1 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset celeba --loss normal --epoch 400 --C_max 50 --gamma 1000 --manualSeed 3  --C_stop_iter 300000 --hinge_loss 1 --save_dir New_celeba_gamma1000_normal_c50_cstep30k_hingeloss &
#DS shujuji zengjia jiange 
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 25 --gamma 1000 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c25_cstop30k_random_add_space1 &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset dsprites --loss bernoulli --epoch 100 --C_max 25 --gamma 1000 --manualSeed 4 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c25_cstop30k_random_add_space3 &

CUDA_VISIBLE_DEVICES=1 python train.py --dataset dsprites --loss bernoulli --epoch 60 --C_max 30 --gamma 100 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c30_cstop30k_random_add_space1 &
CUDA_VISIBLE_DEVICES=2 python train.py --dataset dsprites --loss bernoulli --epoch 100 --C_max 30 --gamma 100 --manualSeed 4 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_ds_gamma1000_ber_c30_cstop30k_random_add_space3 &


CUDA_VISIBLE_DEVICES=1 python train.py --dataset 3dchairs --loss bernoulli --epoch 240 --C_max 0 --C_start 0 --gamma 10 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_3dchairs_betaVAE &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset 3dchairs --loss bernoulli --epoch 240 --C_max 0 --C_start 0 --gamma 1 --manualSeed 3 --C_stop_iter 300000 --hinge_loss 0 --save_dir New_3dchairs_VAE &


