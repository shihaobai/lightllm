# 快速开始

## 安装 lightllm

### 从源码安装
测试环境：Pytorch>=1.3, CUDA 11.8, Python 3.9

```bash
# 创建conda环境（推荐）
# conda create -n lightllm python=3.9 -y

git clone https://github.com/ModelTC/lightllm.git
cd lightllm

pip install -r requirements.txt

python setup.py install
```
本项目的代码在多种GPU上都进行了测试，包括 V100, A100, A800, 4090, 和 H800。

如果你使用 A100 、A800 等显卡，那么推荐你安装 triton==2.1.0 ： 
```bash
pip install triton==2.1.0 --no-deps
``` 

如果你使用 H800、V100 等显卡，那么推荐你安装 triton-nightly，原因请参考：[issue](null) 和 [fix PR](null) ：
```bash
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly --no-deps
```
### 使用docker

使用lightllm最简单的方法是使用官网镜像：

- 第一步：从官网拉取镜像
    ```bash
    docker pull ghcr.io/modeltc/lightllm:main
    ```
- 第二步：运行镜像并开启GPU支持和端口映射
    ```bash
    docker run -it --gpus all -p 8080:8080              \
        --shm-size 1g -v your_local_path:/data/         \
        ghcr.io/modeltc/lightllm:main /bin/bash
    ```
或者你也可以手动构建镜像：
```bash
docker build -t <image_name> .
docker run -it --gpus all -p 8080:8080                  \
        --shm-size 1g -v your_local_path:/data/         \
        <image_name> /bin/bash
```
或者使用```tools/quick_launch_docker.py```辅助脚本，直接启动镜像和服务：
```bash
python tools/quick_launch_docker.py --help
```

> Note: 如果你使用多卡，你也许需要提高上面的 --shm_size 的参数设置。

## 准备模型
lightllm当前支持大部分的主流开源模型，包括大语言模型以及多模态模型，例如：
- [BLOOM](https://huggingface.co/bigscience/bloom)
- [LLaMA](https://github.com/facebookresearch/llama)
- [LLaMA V2](https://huggingface.co/meta-llama)
- [StarCoder](https://github.com/bigcode-project/starcoder)
- [Qwen-7b](https://github.com/QwenLM/Qwen-7B)
- [ChatGLM2-6b](https://github.com/THUDM/ChatGLM2-6B)
- [Baichuan-7b](https://github.com/baichuan-inc/Baichuan-7B)
- [Baichuan2-7b](https://github.com/baichuan-inc/Baichuan2)
- [Baichuan2-13b](https://github.com/baichuan-inc/Baichuan2)    
- [Baichuan-13b](https://github.com/baichuan-inc/Baichuan-13B)
- [InternLM-7b](https://github.com/InternLM/InternLM)
- [Yi-34b](https://huggingface.co/01-ai/Yi-34B)  
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL)
- [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [Llava-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- [Llava-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b)  
- [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [Stablelm](https://huggingface.co/stabilityai/stablelm-2-1_6b)
- [MiniCPM](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16)
- [Phi-3](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3)
- [CohereForAI](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
> 更多支持的模型请查看项目的主页介绍。

下面的内容将会以[Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B)和
[Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)为例，分别演示lightllm对大语言模型以及多模态模型的支持。

下载模型的方法可以参考文章：[如何快速下载huggingface模型——全方法总结](https://zhuanlan.zhihu.com/p/663712983)

可以参考下面的指令下载这两个模型：
> 下面的指令因为网络的原因，可能会花费大量时间，你可以使用上述其它的模型作为替代。
```bash
# mkdirs ~/models && cd ~/models

pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen2-0.5B --local-dir Qwen2-0.5

huggingface-cli download Qwen/Qwen-VL-Chat --local-dir Qwen-VL-Chat

```


## 部署LLM服务

下载完```Qwen2-0.5B```模型以后，在终端使用下面的代码部署API服务：
```bash
# ! 这里的 --model_dir 需要修改成你本机的模型路径
python -m lightllm.server.api_server --model_dir ~/models/Qwen2-0.5B \
                                     --host 0.0.0.0                  \
                                     --port 8080                     \
                                     --tp 1                          \
                                     --max_total_token_num 120000    \
                                     --trust_remote_code
```

服务成功启动后，在另一个终端对API服务进行测试：
```bash
curl http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "What is AI?",
        "parameters":{
            "max_new_tokens":17, 
            "frequency_penalty":1
        }
    }'
```

## 部署VLM服务
下载完```Qwen-VL-Chat```模型以后，在终端使用下面的代码部署API服务：
```bash
# ! 这里的 --model_dir 需要修改成你本机的模型路径
python -m lightllm.server.api_server \
    --host 0.0.0.0                   \
    --port 8080                      \
    --tp 1                           \
    --max_total_token_num 12000      \
    --trust_remote_code              \
    --enable_multimodal              \
    --cache_capacity 1000            \
    --model_dir ~/models/Qwen-VL-Chat
```

服务成功启动后，使用如下的python代码对API服务进行测试：
```python
import json
import requests
import base64

def run(query, uris):
    images = []
    for uri in uris:
        if uri.startswith("http"):
            images.append({"type": "url", "data": uri})
        else:
            with open(uri, 'rb') as fin:
                b64 = base64.b64encode(fin.read()).decode("utf-8")
            images.append({'type': "base64", "data": b64})

    data = {
        "inputs": query,
        "parameters": {
            "max_new_tokens": 200,
            # The space before <|endoftext|> is important, 
            # the server will remove the first bos_token_id, 
            # but QWen tokenizer does not has bos_token_id
            "stop_sequences": [" <|endoftext|>", " <|im_start|>", " <|im_end|>"],
        },
        "multimodal_params": {
            "images": images,
        }
    }

    url = "http://127.0.0.1:8080/generate"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response

query = """
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<img></img>
这是什么？<|im_end|>
<|im_start|>assistant
"""

response = run(
    uris = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    ],
    query = query
)

if response.status_code == 200:
    print(f"Result: {response.json()}")
else:
    print(f"Error: {response.status_code}, {response.text}")
```

## 下一步

### 更好地使用lightllm

- [模型部署参数详解](NULL)
- [API调用参数详解](NULL)
- [lightllm性能测试](NULL)

### 更好地了解lightllm

- [lightllm的架构设计](NULL)
- [TokenAttention](NULL)

### 为lightllm贡献代码
- [lightllm添加新模型支持](NULL)
