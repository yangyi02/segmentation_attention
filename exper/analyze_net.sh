GPU_ID=0

TOOLS=../code_baidu/build/tools
PYTHON=../code_baidu/python

DATASET=voc12
CONFIG=config
NET_ID=vgg128_noup_pool3_20M_largewin_multipath
PROTO_MODEL=${DATASET}/${CONFIG}/${NET_ID}/train_train_aug.prototxt

MODEL=${DATASET}/model/${NET_ID}/init.caffemodel

FIG_DIR=${DATASET}/fig/${NET_ID}
mkdir -p ${FIG_DIR}
LOG_DIR=${DATASET}/log/${NET_ID}
mkdir -p ${LOG_DIR}

# --------------------
# Draw caffe network
echo "Draw the network ..."
${PYTHON}/draw_net.py ${PROTO_MODEL} ${FIG_DIR}/model.png

# --------------------
# Check initialized model weights
echo "Check initialization ..."
${TOOLS}/caffe init --model=${PROTO_MODEL} --weights=${MODEL} --gpu=${GPU_ID} 2>&1 | tee ${LOG_DIR}/init.log
##echo "Check time spending ..."
#${TOOLS}/caffe time --model=${PROTO_MODEL} --gpu=${GPU_ID} 2>&1 | tee ${LOG_DIR}/time.log
