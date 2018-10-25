#!bin/bash

id=(2011_002675 2011_002322 2011_001988 2011_001060 2011_000900 2011_000248 2007_000323 2009_001854 2007_000799 2007_002728 2007_003188 2007_003503 2007_001311 2007_001630 2007_005173 2007_005331 2007_009084 2008_003461 2008_007945 2008_008221 2009_000457 2010_001024)

#id=(2007_000491 2007_000783 2007_001568 2007_001761 2007_001884 2007_002046 2007_006802 2007_008260 2010_005888 2010_005344 2010_005284 2010_005108 2010_004795 2010_004789 2010_004628 2010_004322 2010_004320 2010_004120 2011_000813 2011_000536 2011_000070 2011_001232 2011_001341 2011_001642 2011_002098 2011_002121 2011_002343 2011_002589 2011_002675 2011_003256 2011_003271)

dataset=voc12
baseline=vgg128_noup_pool3_20M_largewin3_coco
model=vgg128_noup_pool3_20M_largewin_coco_attention13
#baseline=vgg128_noup_pool3_20M_largewin3
#model=vgg128_noup_pool3_20M_largewin_attention4

for ii in ${id[@]}
do
    cp /home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/${dataset}/res/features/${baseline}/val/fc8/post_none/results/VOC2012/Segmentation/comp6_val_cls/${ii}.png ./res_baseline
    cp /home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/${dataset}/res/features/${model}/val/fc8/post_none/results/VOC2012/Segmentation/comp6_val_cls/${ii}.png ./res_sharenet
    cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/${ii}.jpg ./img
    cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/${ii}.png ./gt
done


# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2009_001854.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2009_001854.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_000799.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_000799.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_002728.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_002728.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_003188.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_003188.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_003503.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_003503.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_001311.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_001311.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_001630.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_001630.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_005173.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_005173.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_005331.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_005331.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2007_009084.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2007_009084.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003461.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2008_003461.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_007945.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2008_007945.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_008221.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2008_008221.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2009_000457.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2009_000457.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2010_001024.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationClass/2010_001024.png ./gt
