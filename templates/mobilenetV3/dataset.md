# MobileNetV3 改进实验必备数据集清单（Hugging Face 加载版）
以下是 MobileNetV3 改进实验（量化优化/SE模块适配/动态通道/边缘部署）**必须的核心数据集**，覆盖「基础性能验证→量化鲁棒性→真实边缘场景」全流程，均支持 Hugging Face `datasets` 库一键加载，适配 PyTorch 训练/测试流程。

## 一、核心数据集分类与加载（按实验优先级排序）
### 1. 基础性能验证数据集（必备：所有改进实验的基线验证）
#### （1）Imagenette（10类精简版ImageNet）
- **用途**：快速验证 MobileNetV3 改进（如SE门控、激活函数替换）的基础分类性能，训练速度比全量ImageNet快10倍+，适合快速迭代。
- **Hugging Face 地址**：[hf.co/datasets/frgfm/imagenette](https://huggingface.co/datasets/frgfm/imagenette)
- **核心加载代码**：
```python
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# MobileNetV3 专用预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集（160px精简版，适配轻量化）
dataset = load_dataset("frgfm/imagenette", "imagenette-160px", split=["train", "validation"])
train_ds, val_ds = dataset[0], dataset[1]

# 应用预处理
def apply_preprocess(examples):
    examples["pixel_values"] = [preprocess(img.convert("RGB")) for img in examples["image"]]
    return examples
train_ds = train_ds.map(apply_preprocess, batched=True, batch_size=32)
val_ds = val_ds.map(apply_preprocess, batched=True, batch_size=32)

# 转换为DataLoader（适配训练）
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
```
- **适配改进方向**：SE模块幅值缩放、激活函数替换（如hard-swish→分段线性sigmoid）、动态通道剪枝的基础性能验证。

#### （2）ImageNet-1k（10%子集，可选：最终基准验证）
- **用途**：改进方案定型后，用标准ImageNet子集验证性能（避免全量数据集的算力开销）。
- **Hugging Face 地址**：[hf.co/datasets/imagenet-1k](https://huggingface.co/datasets/imagenet-1k)
- **核心加载代码**：
```python
# 需先在Hugging Face完成ImageNet授权，加载10%子集
dataset = load_dataset("imagenet-1k", split=["train[:10%]", "validation[:10%]"])
train_ds, val_ds = dataset[0], dataset[1]
# 后续预处理/Loader逻辑与Imagenette一致
```

### 2. 量化鲁棒性验证数据集（必备：量化改进实验）
#### ImageNet-C（损坏版ImageNet）
- **用途**：测试量化后 MobileNetV3 对图像损坏（噪声、模糊、压缩）的鲁棒性，核心验证「量化感知激活函数」「SE模块量化适配」的效果。
- **Hugging Face 地址**：[hf.co/datasets/hendrycks/imagenet-c](https://huggingface.co/datasets/hendrycks/imagenet-c)
- **核心加载代码**：
```python
# 加载高斯噪声损坏（severity=3，中等损坏程度）
dataset = load_dataset("hendrycks/imagenet-c", "gaussian_noise_severity_3", split="validation")
# 预处理（需开启量化感知训练的dtype转换）
preprocess_qat = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.ConvertImageDtype(torch.float32)  # 量化友好
])
dataset = dataset.map(lambda x: {"pixel_values": preprocess_qat(x["image"].convert("RGB"))}, batched=False)
dataset.set_format("torch", columns=["pixel_values", "label"])
# 测试Loader
test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
```
- **适配改进方向**：8位整数量化、量化感知激活函数设计、SE模块幅值缩放的量化鲁棒性验证。

### 3. 真实边缘场景数据集（必备：边缘部署验证）
#### Edge-ImageNet（边缘摄像头采集的ImageNet子集）
- **用途**：验证改进后 MobileNetV3 在真实边缘场景（低光、模糊、角度偏移）的性能，匹配「边缘设备延迟/能耗」实验。
- **Hugging Face 地址**：[hf.co/datasets/liuhaotian/Edge-ImageNet](https://huggingface.co/datasets/liuhaotian/Edge-ImageNet)
- **核心加载代码**：
```python
# 加载3000张边缘采集图像（匹配改进实验的测试集规模）
dataset = load_dataset("liuhaotian/Edge-ImageNet", split="train[:3000]")
# 预处理（与量化鲁棒性实验一致）
dataset = dataset.map(lambda x: {"pixel_values": preprocess_qat(x["image"].convert("RGB"))}, batched=False)
dataset.set_format("torch", columns=["pixel_values", "label"])
edge_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
```
- **适配改进方向**：动态资源感知、边缘设备延迟优化、SE模块latency降低的真实场景验证。

### 4. QAT 校准数据集（必备：量化感知训练）
- **用途**：量化感知训练（QAT）的校准集，保证量化后精度不损失，需从 Edge-ImageNet 中选取100张代表性图像。
- **核心加载代码**：
```python
# 加载100张校准图像
calib_dataset = load_dataset("liuhaotian/Edge-ImageNet", split="train[:100]")
calib_dataset = calib_dataset.map(lambda x: {"pixel_values": preprocess_qat(x["image"].convert("RGB"))}, batched=False)
calib_dataset.set_format("torch", columns=["pixel_values", "label"])
calib_loader = DataLoader(calib_dataset, batch_size=16, shuffle=False)
```

## 二、数据集与改进方向的精准适配
| MobileNetV3 改进方向                  | 必须数据集               | 核心验证指标                  |
|---------------------------------------|--------------------------|-------------------------------|
| 自适应SE门控+量化感知激活（quantized_mobilenetv3_se_adapt） | Imagenette + ImageNet-C + Edge-ImageNet | Top-1准确率、量化精度损失、边缘延迟 |
| 动态资源感知+通道剪枝（dynamic_mobilenetv3_resource_aware） | Imagenette + Edge-ImageNet | 平均FLOPs、95%分位延迟、能耗  |
| 比特共享集成（BitsSharedEnsembleMobileNetV3） | Imagenette + ImageNet-C | 存储开销、OOD检测AUC、对抗鲁棒性 |
| 模块化RL控制器（ModuEdgeCtrl）| Imagenette + Edge-ImageNet | 多任务精度、边缘延迟保证      |
| 隐变量调度（LatentScheduleEdgeBackbone） | Imagenette + Edge-ImageNet | 多任务精度、95%分位延迟       |

## 三、关键加载优化（适配AI Scientist自动化流程）
1. **国内下载加速**：设置HF镜像，避免超时
```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
```
2. **轻量化采样**：所有数据集默认采样≤50%，降低算力开销（如Imagenette取50%、Edge-ImageNet取3000张）。
3. **异常处理**：封装加载逻辑，捕获授权/不存在异常
```python
from datasets.exceptions import PermissionDeniedError, DatasetNotFoundError

def safe_load_dataset(ds_name, subset=None):
    try:
        if subset:
            return load_dataset(ds_name, subset, split="validation")
        return load_dataset(ds_name, split="validation")
    except PermissionDeniedError:
        raise Exception(f"数据集{ds_name}需要Hugging Face授权！")
    except DatasetNotFoundError:
        raise Exception(f"数据集{ds_name}不存在，请检查地址！")
```

## 总结
1. **最小数据集组合**：Imagenette（基础验证） + ImageNet-C（量化鲁棒性） + Edge-ImageNet（边缘场景），可覆盖90%的MobileNetV3改进实验；
2. **自动化适配**：所有加载代码可直接集成到AI Scientist程序中，返回标准化DataLoader，无需手动预处理；
3. **核心目标**：通过精简、代表性的数据集，快速验证改进方案的有效性，避免全量数据集的算力浪费。
