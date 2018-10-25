#!bin/bash

#id=(000000000136 000000000192 000000000241 000000000328 000000000395 000000000428 000000000459 000000000488 000000000536 000000000599 000000000623 000000000632 000000000641 000000000692 000000000757 000000000761 000000000923 000000000974 000000001000 000000001153 000000001164 000000001228 000000001268 000000001532 000000001584 000000001626 000000001840 000000001869 000000002014 000000002235 000000002239)

id=(000000000042 000000000294 000000000675 000000000923 000000000985 000000001089 000000001146 000000001532 000000001592 000000001626 000000001799 000000002235 000000002255 000000002562 000000002972 000000002985 000000003109 000000005107 000000006847 000000006896 000000007088 000000007115 000000007304 000000007682 000000009007 000000010056 000000010123 000000010205 000000010693 000000011202 000000012192 000000012748 000000015301 000000016285 000000016977 000000018444 000000018462 000000020254 000000020774)

dataset=coco
baseline=vgg128_noup_pool3_20M_largewin
model=vgg128_noup_pool3_20M_largewin_attention12

for ii in ${id[@]}
do
    cp /home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/${dataset}/res/features/${baseline}/val/fc8/post_none/results/COCO2014/Segmentation/comp6_val_cls/COCO_val2014_${ii}.png ./res_baseline
    cp /home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/${dataset}/res/features/${model}/val/fc8/post_none/results/COCO2014/Segmentation/comp6_val_cls/COCO_val2014_${ii}.png ./res_sharenet
    cp /home/lcchen/workspace/rmt/data/coco/JPEGImages/COCO_val2014_${ii}.jpg ./img
    cp /home/lcchen/workspace/rmt/data/coco/SegmentationClass/COCO_val2014_${ii}.png ./gt
done


