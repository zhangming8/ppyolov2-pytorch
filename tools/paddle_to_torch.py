import torch
import paddle
import sys

sys.path.append(".")
from models.backbone import ResNetVd


def convert_paddle_weight(torch_model, pretrain_paddle_weight):
    torch_model_dict = torch_model.state_dict()
    paddle_params = paddle.load(pretrain_paddle_weight)
    torch_params = dict()

    for k in sorted(list(paddle_params.keys())):
        # print(k)
        k_new = k.replace("backbone.", '').replace("_mean", "running_mean").replace("_variance", "running_var")
        torch_params[k_new] = torch.from_numpy(paddle_params[k])

    for k in sorted(list(torch_model_dict.keys())):
        if "num_batches_tracked" in str(k):
            continue
        assert k in torch_params.keys(), "error key {}".format(k)
        assert torch_params[k].shape == torch_model_dict[k].shape
        # print(torch_params[k].dtype)

    torch_model.load_state_dict(torch_params, strict=True)
    print('load weights from paddle {}'.format(pretrain_paddle_weight))
    torch.save(model.state_dict(), save_weight)
    print("==>> save converted model to {}".format(save_weight))


if __name__ == "__main__":
    # paddle ResNet-vd weight https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_vd_ssld_pretrained.pdparams
    paddle_weight = "../weights/ResNet50_vd_ssld_pretrained.pdparams"
    # paddle_weight = "../weights/ResNet50_vd_ssld_v2_pretrained.pdparams"
    # pytorch ResNet-vd
    model = ResNetVd(depth=50, variant='d', out_indices=[1, 2, 3], dcn_v2_stages=[-1])

    save_weight = paddle_weight.replace(".pdparams", ".pth")
    convert_paddle_weight(model, paddle_weight)
