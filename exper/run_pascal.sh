#!/bin/sh

# MODIFY PATH for YOUR SETTING
ROOT_DIR=/home/lcchen/workspace

CAFFE_DIR=../code
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

#EXP=voc12
#EXP=voc10_part
EXP=cloth_data_6
#EXP=voc10_person

if [ "${EXP}" = "voc12" ]; then
    NUM_LABELS=21
    DATA_ROOT=${ROOT_DIR}/rmt/data/pascal/VOCdevkit/VOC2012
elif [ "${EXP}" = "voc10_part" ]; then
    NUM_LABELS=7
    DATA_ROOT=${ROOT_DIR}/rmt/data/pascal/VOCdevkit/VOC2012
elif [ "${EXP}" = "voc10_person" ]; then
    NUM_LABELS=2
    DATA_ROOT=${ROOT_DIR}/rmt/data/pascal/VOCdevkit/VOC2012
elif [ "${EXP}" = "cloth_data_6" ]; then
    NUM_LABELS=6
    DATA_ROOT=${ROOT_DIR}/rmt/data/cloth_data_6/
else
    NUM_LABELS=0
    echo "Wrong exp name"
fi
 

# Specify which model to train
################ voc10_part ##############
#NET_ID=vgg128_noup_pool3_20M_largewin
#NET_ID=vgg128_noup_pool3_29M_largewin
#NET_ID=vgg128_noup_pool3_20M_largewin2
#NET_ID=vgg128_noup_pool3_20M_largewin3
#NET_ID=deconv_exp1
#NET_ID=deconv_exp2
#NET_ID=deconv_exp3
#NET_ID=deconv_exp4
#NET_ID=deconv_exp5
#NET_ID=deconv_exp6
#NET_ID=deconv_exp7
#NET_ID=deconv_exp8
#NET_ID=deconv_exp9
#NET_ID=deconv_exp10
#NET_ID=deconv_exp11
#NET_ID=deconv_exp12
#NET_ID=deconv_exp13
#NET_ID=deconv_exp14
#NET_ID=deconv_exp15
#NET_ID=deconv_exp16
#NET_ID=deconv_ms1
#NET_ID=deconv_ms5
#NET_ID=deconv_exp17
#NET_ID=vgg128_ms_pool3_20M_largewin
#NET_ID=deconv_ms6
#NET_ID=deconv_ms7
#NET_ID=deconv_ms8
#NET_ID=deconv_ms9
#NET_ID=deconv_exp18
#NET_ID=deconv_ms10
#NET_ID=deconv_ms12
#NET_ID=deconv_exp19
#NET_ID=vgg128_noup_pool3_20M_largewin_attention
#NET_ID=vgg128_noup_pool3_20M_largewin_attention2
#NET_ID=vgg128_noup_pool3_20M_largewin_attention3
#NET_ID=vgg128_noup_pool3_20M_largewin_attention4
#NET_ID=vgg128_noup_pool3_20M_largewin_attention5
#NET_ID=vgg128_noup_pool3_20M_largewin_attention6
#NET_ID=vgg128_noup_pool3_20M_largewin_attention7
#NET_ID=vgg128_noup_pool3_20M_largewin_attention8
#NET_ID=vgg128_noup_pool3_20M_largewin_attention9
#NET_ID=vgg128_noup_pool3_20M_largewin_attention10
#NET_ID=vgg128_noup_pool3_20M_largewin_attention11
#NET_ID=vgg128_noup_pool3_20M_largewin_attention12
#NET_ID=vgg128_noup_pool3_20M_largewin_attention13
#NET_ID=vgg128_noup_pool3_20M_largewin_attention14
#NET_ID=vgg128_noup_pool3_20M_largewin_attention15
#NET_ID=vgg128_noup_pool3_20M_largewin_attention16
#NET_ID=vgg128_noup_pool3_20M_largewin_attention17
#NET_ID=vgg128_noup_pool3_20M_largewin_attention18
#NET_ID=vgg128_noup_pool3_20M_largewin_attention19
#NET_ID=vgg128_noup_pool3_20M_largewin_attention20
#NET_ID=vgg128_noup_pool3_20M_largewin_attention21
#NET_ID=vgg128_noup_pool3_20M_largewin_attention22
#NET_ID=vgg128_noup_pool3_20M_largewin_attention23
#NET_ID=vgg128_noup_pool3_20M_largewin_attention24
#NET_ID=vgg128_noup_pool3_20M_largewin_attention25
#NET_ID=vgg128_noup_pool3_20M_largewin_attention26
#NET_ID=vgg128_noup_pool3_20M_largewin_attention27
#NET_ID=vgg128_noup_pool3_20M_largewin_attention28
#NET_ID=vgg128_ms_pool3_20M_largewin_attention
#NET_ID=vgg128_ms_pool3_20M_largewin2
#NET_ID=vgg128_noup_pool3_20M_largewin4
#NET_ID=vgg128_ms_pool3_20M_largewin5
#NET_ID=vgg128_noup_pool3_20M_largewin_attention31
#NET_ID=vgg128_noup_pool3_20M_largewin_attention38
#NET_ID=vgg128_ms_pool3_20M_largewin_attention19
#NET_ID=vgg128_noup_pool3_20M_largewin_attention40
#NET_ID=vgg128_ms_pool3_20M_largewin_attention20
#NET_ID=vgg128_noup_pool3_20M_largewin_attention21_2
#NET_ID=vgg128_ms_pool3_20M_largewin_attention8_2
#NET_ID=vgg128_noup_pool3_20M_largewin_attention45_2
#NET_ID=vgg128_ms_pool3_20M_largewin_attention31
#NET_ID=vgg128_noup_pool3_20M_largewin_attention46
#NET_ID=vgg128_ms_pool3_20M_largewin_attention30

########### voc12 ################
#NET_ID=vgg128_ms_pool3_20M_largewin3_coco

########### cloth_data_6 ################
#NET_ID=vgg128_noup_pool3_20M_largewin
NET_ID=vgg128_noup_pool3_20M_largewin_attention1

########### voc10_person #########
#NET_ID=deconv_exp1
#NET_ID=deconv_exp2
#NET_ID=deconv_exp3
#NET_ID=vgg128_noup_pool3_20M_largewin

#TRAIN_SET_SUFFIX=
#TRAIN_SET_SUFFIX=_aug

#TRAIN_SET_STRONG=train
#TRAIN_SET_STRONG=train200
#TRAIN_SET_STRONG=train500
#TRAIN_SET_STRONG=train1000
#TRAIN_SET_STRONG=train750

#TRAIN_SET_WEAK_LEN=5000

DEV_ID=1

#####

# Create dirs

CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

# Run

RUN_TRAIN=1
RUN_TEST=1
RUN_TRAIN2=0
RUN_TEST2=0
RUN_SAVE=0

# Training #1 (on train_aug)

if [ ${RUN_TRAIN} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=train${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
	TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
	comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
	TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
	comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init.caffemodel
    #
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
	if [ -f ${MODEL} ]; then
	    CMD="${CMD} --weights=${MODEL}"
	fi
        # resume
        #--snapshot=${EXP}/model/${NET_ID}/train_iter_1000.solverstate \
	echo Running ${CMD} && ${CMD}
fi

# Test #1 specification (on val or test)

if [ ${RUN_TEST} -eq 1 ]; then
    #
    for TEST_SET in val; do
	TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
	MODEL=${EXP}/model/${NET_ID}/test.caffemodel
	#MODEL=${EXP}/model/${NET_ID}/train_iter_4000.caffemodel
	if [ ! -f ${MODEL} ]; then
	    MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
	fi
	#
	echo Testing net ${EXP}/${NET_ID}
	FEATURE_DIR=${EXP}/features/${NET_ID}
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc9
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
	CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
	echo Running ${CMD} && ${CMD}
    done
fi

# Training #2 (finetune on trainval_aug)

if [ ${RUN_TRAIN2} -eq 1 ]; then
    #
    LIST_DIR=${EXP}/list
    TRAIN_SET=trainval${TRAIN_SET_SUFFIX}
    if [ -z ${TRAIN_SET_WEAK_LEN} ]; then
	TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}
	comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    else
	TRAIN_SET_WEAK=${TRAIN_SET}_diff_${TRAIN_SET_STRONG}_head${TRAIN_SET_WEAK_LEN}
	comm -3 ${LIST_DIR}/${TRAIN_SET}.txt ${LIST_DIR}/${TRAIN_SET_STRONG}.txt | head -n ${TRAIN_SET_WEAK_LEN} > ${LIST_DIR}/${TRAIN_SET_WEAK}.txt
    fi
    #
    MODEL=${EXP}/model/${NET_ID}/init2.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    #
    echo Training2 net ${EXP}/${NET_ID}
    for pname in train solver2; do
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
        CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID}"
	echo Running ${CMD} && ${CMD}
fi

# Test #2 on official test set

if [ ${RUN_TEST2} -eq 1 ]; then
    #
    for TEST_SET in val test; do
	TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
	MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
	if [ ! -f ${MODEL} ]; then
	    MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
	fi
	#
	echo Testing2 net ${EXP}/${NET_ID}
	FEATURE_DIR=${EXP}/features2/${NET_ID}
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
	mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
	sed "$(eval echo $(cat sub.sed))" \
	    ${CONFIG_DIR}/test.prototxt > ${CONFIG_DIR}/test_${TEST_SET}.prototxt
	CMD="${CAFFE_BIN} test \
             --model=${CONFIG_DIR}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
	echo Running ${CMD} && ${CMD}
    done
fi

# Translate and save the model

if [ ${RUN_SAVE} -eq 1 ]; then
    #
    MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
    if [ ! -f ${MODEL} ]; then
	MODEL=`ls -t ${EXP}/model/${NET_ID}/train*_iter_*.caffemodel | head -n 1`
    fi
    MODEL_DEPLOY=${EXP}/model/${NET_ID}/deploy.caffemodel
    #
    echo Translating net ${EXP}/${NET_ID}
        CMD="${CAFFE_BIN} save \
         --model=${CONFIG_DIR}/deploy.prototxt \
         --weights=${MODEL} \
         --out_weights=${MODEL_DEPLOY}"
	echo Running ${CMD} && ${CMD}
fi

