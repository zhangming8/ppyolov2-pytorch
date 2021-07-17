# Pytorch reimplement of PPYOLOv2

# environment
    pytorch>=1.7.0, python>=3.6

# TODO
    add p6, p7 head
    release pretrain weight

# DONE
    EMA
    DropBlock
    IoULoss
    IouAware
    GridSensitive
    SPP
    grad_clip
    PAN neck
    FPN neck
    auto AMP
    multi-size training
    MixUp
    GridMask
    mosaic
    ResNext, res2net, CSPdarknet backbone
    ResNet-vd backbone and add DCN in stage3
    warmup
    accumulate grad to increace batch size

# train
    sh train.sh
    
# evaluate
    sh evaluate.sh
    
# predict/inference/demo
    sh predict.sh
   

# reference
    https://github.com/PaddlePaddle/PaddleDetection
    https://github.com/open-mmlab/mmdetection
    https://github.com/JDAI-CV/fast-reid
    https://github.com/ultralytics/YOLOv5
    https://github.com/xingyizhou/CenterNet
