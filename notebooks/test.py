# %%
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import onnxruntime as ort
from PIL import Image

from src.models.components.backbone import EfficientNetB3
from typing import List, Dict, Tuple, Union


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 128
        self.resize = transforms.Resize(self.image_size)

        feature_size = 256
        self.hidden_ch = [128, 128, 64, 32]

        self.backbone = EfficientNetB3(out_ch=feature_size)

        image_ch = 3

        self.linear1 = nn.Conv2d(feature_size, self.hidden_ch[0], kernel_size=1)
        self.conv1 = nn.Conv2d(image_ch, self.hidden_ch[0], kernel_size=1)
        self.linear2 = nn. Conv2d(self.hidden_ch[0], self.hidden_ch[1], kernel_size=1)
        self.conv2 = nn.Conv2d(self.hidden_ch[0], self.hidden_ch[1], kernel_size=1)
        self.linear3 = nn.Conv2d(self.hidden_ch[1], self.hidden_ch[2], kernel_size=1)
        self.conv3 = nn.Conv2d(self.hidden_ch[1], self.hidden_ch[2], kernel_size=1)
        self.linear4 = nn. Conv2d(self.hidden_ch[2], self.hidden_ch[3], kernel_size=1)
        self.conv4 = nn.Conv2d(self.hidden_ch[2], self.hidden_ch[3], kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.conv_final = nn.Conv2d(in_channels=self.hidden_ch[3], out_channels=3, kernel_size=1)

    def forward(self, patch, x_resized):
        feature = self.backbone(x_resized)

        images = patch

        f = feature.view(feature.size(0), -1, 1, 1)  # ブロードキャストできるようにshapeを変換
        f = self.linear1(f)
        patch = self.conv1(patch) + f
        patch = self.relu(patch)

        # second
        f = self.linear2(f)
        patch = self.conv2(patch) + f
        patch = self.relu(patch)

        # third
        f = self.linear3(f)
        patch = self.conv3(patch) + f
        patch = self.relu(patch)

        # forth
        f = self.linear4(f)
        patch = self.conv4(patch) + f
        patch = self.relu(patch)

        patch = self.conv_final(patch)
        patch = patch + images

        patch = torch.clamp(patch, 0, 1)  # (B, C, H, W)

        return patch

# %%
# max_val = 4
# model = Net()
# image_size = 128
# resize = transforms.Resize((image_size, image_size))
# model.eval()

# values = torch.linspace(0, 1, max_val+1)
# combinations = torch.cartesian_prod(values, values, values).unsqueeze(2).unsqueeze(3)

# x = Image.open("/workspace/data/validation/GT_IMAGES/a4674-_DSC0089.jpg").convert("RGB")

# patch = transforms.ToTensor()(x).unsqueeze(0)

# # ダミーの入力データを生成 (combinationsとpatchの形状を使用)
# dummy_combinations = combinations  # take only 1 for dummy
# dummy_patch = resize(patch)

# # ONNX形式でエクスポート
# torch.onnx.export(model, (dummy_combinations, dummy_patch), f"cache_model_{max_val}.onnx")
# %%




def infer_model(combinations, patch):
    return model(combinations, resize(patch))




# def infer_onnx_model(combinations, patch):
#     # 入力データを辞書形式で渡す
#     ort_inputs = {ort_session.get_inputs()[0].name: combinations.cpu().numpy(),
#                   ort_session.get_inputs()[1].name: patch.cpu().numpy()}
#     ort_outs = ort_session.run(None, ort_inputs)
#     return torch.from_numpy(ort_outs[0])

# def quantize(rgb_value, max_val):
#     scale_factor = 1 / max_val
#     quantized_value = (rgb_value / scale_factor).round() * scale_factor
#     print(rgb_value / scale_factor)
#     print((rgb_value / scale_factor).round().unique())
#     return quantized_value

# without 1
def quantize(rgb_value, max_val):
    scale_factor = 1 / max_val
    quantized_value = (rgb_value / scale_factor).floor() * scale_factor
    return quantized_value

def build_cache(outputs: torch.Tensor, combinations: torch.Tensor, max_val: int) -> torch.Tensor: 
    cache_size: int = max_val * max_val * max_val
    cache: torch.Tensor = torch.zeros((cache_size, 3, outputs.shape[2], outputs.shape[3]))

    # ベクトル化されたインデックス計算
    r_int = (combinations[:, 0, 0, 0] * max_val).int()
    g_int = (combinations[:, 1, 0, 0] * max_val).int()
    b_int = (combinations[:, 2, 0, 0] * max_val).int()

    indices = r_int * max_val**2 + g_int * max_val + b_int
    print(indices.unique())
    # インデックスを使用してcacheを更新
    cache[indices, :, :, :] = outputs.view(-1, 3, 1, 1)

    return cache


def create_output(quantized_patch: torch.Tensor, cache: torch.Tensor, max_val: int) -> torch.Tensor:
    r_int = (quantized_patch[0, 0] * (max_val)).int()
    g_int = (quantized_patch[0, 1] * (max_val)).int()
    b_int = (quantized_patch[0, 2] * (max_val)).int()
    index = r_int * max_val**2 + g_int * max_val + b_int
    print(index.unique())

    # Reshape the index tensor to match the expected shape for advanced indexing with cache
    index = index.reshape(-1)

    # Use advanced indexing to retrieve values from cache
    output = cache[index]

    # Reshape the output tensor back to (1, 3, height, width)
    output = output.reshape(1, 3, quantized_patch.shape[2], quantized_patch.shape[3])

    return output

def process_patch(patch, cache, max_val: int):
    quantized_patch = quantize(patch, max_val)
    return create_output(quantized_patch, cache, max_val)



if __name__ == "__main__":
    import time

    max_val = 2
    # ort_session = ort.InferenceSession(f"/cache_model_{max_val}.onnx", providers=["CPUExecutionProvider"])

    x = Image.open("/workspace/data/validation/GT_IMAGES/a4674-_DSC0089.jpg")

    patch = transforms.ToTensor()(x).unsqueeze(0)

    model = Net()
    image_size = 128
    resize = transforms.Resize((image_size, image_size))

    ori_size = transforms.Resize((image_size*5, image_size*5))

    patch = ori_size(patch)

    start_time = time.time()
    values = torch.linspace(0, 1, max_val+1)[:-1]
    combinations = torch.cartesian_prod(values, values, values).unsqueeze(2).unsqueeze(3)
    elapsed_time = time.time() - start_time
    print(f"Time for generating combinations: {elapsed_time:.2f} seconds")

    start_time = time.time()
    # outputs = infer_onnx_model(combinations, resize(patch))
    outputs = infer_model(combinations, resize(patch))
    elapsed_time = time.time() - start_time
    print(f"Time for inference: {elapsed_time:.2f} seconds")

    start_time = time.time()
    cache = build_cache(outputs, combinations, max_val)
    elapsed_time = time.time() - start_time
    print(f"Time for building cache: {elapsed_time:.2f} seconds")

    start_time = time.time()
    result = process_patch(patch, cache, max_val)
    elapsed_time = time.time() - start_time
    print(f"Time for processing patch: {elapsed_time:.2f} seconds")
    # print(result)

    from torchvision.transforms.functional import to_pil_image

    # Convert the tensor to PIL Image
    # result_pil = to_pil_image(result.squeeze(0))  # Squeeze to remove the batch dimension

    # model = Net()
    # result = model(patch, resize(patch))

    # result = torch.from_numpy(result)
    output_tensor = result.squeeze(0)
    to_pil = transforms.ToPILImage()
    output_image = to_pil(output_tensor)
    output_image.save(f"test.jpg")
    # Show the PIL Image
    # result_pil.save(f"test.jpg")

# %%
# from PIL import Image
# import torch

# x = Image.open("{}").convert("RGB")

# #x = torch.randn(1, 3, size+35, size+13, dtype=torch.float32)
# x = transforms.ToTensor()(x).unsqueeze(0)
# #x = torch.ones(1, 3, 100, 100, dtype=torch.float32) /2
# #x = torch.randn(1, 3, 600, 600, dtype=torch.float32)
# c = RefinedPixelwiseCachedNet().eval()
# m = Net().eval()
# # %%
# import torch

# model = Net().eval()
# size = 601
# x = torch.randn(1, 3, size, size, dtype=torch.float32)

# torch.onnx.export(model,
#                   x,
#                   f"{size}.onnx",
#                   input_names=["img"],
#                   output_names=["output"],
#                   dynamic_axes={"img": {2: "height", 3: "width"}, 'output': {2: "height", 3: "width"}})

# %%
# import onnxruntime
# import numpy as np
# from torchvision.transforms import transforms
# from PIL import Image

# sess_options = onnxruntime.SessionOptions()
# sess_options.enable_profiling = True

# session = onnxruntime.InferenceSession(f"{size}.onnx", providers=["CPUExecutionProvider"], sess_options=sess_options)

# x = Image.open("/workspace/data/validation/GT_IMAGES/a0398-IMG_5829.jpg").convert("RGB")

# #x = torch.randn(1, 3, size+35, size+13, dtype=torch.float32)
# x = transforms.ToTensor()(x).unsqueeze(0)
# y = x.detach().numpy()
# outs = session.run(None, {"img": y})


# prof_file = session.end_profiling()
# print(prof_file)
# # %%
# t = model(x)
# print(np.mean(abs(t.detach().numpy() - outs[0]))*255)
# print(torch.mean(abs(t-torch.from_numpy(outs[0])))*255)
# # %%
# to_pil = transforms.ToPILImage()
# output_image = to_pil(t.squeeze(0))
# output_image.show()

# output_tensor = torch.from_numpy(outs[0])
# output_tensor = output_tensor.squeeze(0)
# to_pil = transforms.ToPILImage()
# output_image = to_pil(output_tensor)
# output_image.show()
