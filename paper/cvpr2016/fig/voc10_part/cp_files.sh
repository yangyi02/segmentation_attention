#!bin/bash

#id=(2008_000034 2008_000579 2008_000215 2008_000492 2008_000691 2008_002789 2008_003136 2008_003228 2008_003344 2008_003514 2008_003610 2008_003825 2008_005732 2008_005884 2008_006148 2008_007585 2008_000473 2008_000481 2008_000510 2008_000522 2008_000662)
id=(2010_004597 2010_003983 2010_003632 2010_003630 2010_003628 2010_002929 2010_002927 2010_002510 2010_005293 2010_005410 2010_005626 2010_005654 2010_005141 2010_004952 2010_004786)
dataset=voc10_part
baseline=vgg128_noup_pool3_20M_largewin4
model=vgg128_noup_pool3_20M_largewin_attention47

for ii in ${id[@]}
do
    cp /home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/${dataset}/res/features/${baseline}/val/fc8/post_none/results/VOC2010_part/Segmentation/comp6_val_cls/${ii}.png ./res_baseline
    cp /home/lcchen/workspace/rmt/work/deeplabel_baidu/exper/${dataset}/res/features/${model}/val/fc8/post_none/results/VOC2010_part/Segmentation/comp6_val_cls/${ii}.png ./res_sharenet
    cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/${ii}.jpg ./img
    cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/${ii}.png ./gt
done

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000034.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000034.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000579.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000579.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000215.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000215.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000492.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000492.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000691.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000691.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_002789.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_002789.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003136.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_003136.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003228.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_003228.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003344.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_003344.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003514.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_003514.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003610.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_003610.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_003825.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_003825.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_005732.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_005732.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_005884.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_005884.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_006148.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_006148.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_007585.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_007585.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000473.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000473.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000481.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000481.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000510.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000510.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000522.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000522.png ./gt

# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages/2008_000662.jpg ./img
# cp /home/lcchen/workspace/rmt/data/pascal/VOCdevkit/VOC2012/SegmentationPart_Visualization/2008_000662.png ./gt
