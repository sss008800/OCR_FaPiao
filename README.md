# OCR_FaPiao
## used paddleocr infer 增值税发票

本次需求针对增值税发票关键信息识别，采用百度开源的增值税识别模型效果比较差，故自行标注了25张发票进行评估，hmean识别率在93%以上。

### 原图如下：
<img src="https://github.com/sss008800/OCR_FaPiao/assets/16535855/1916479f-4d39-401e-9844-666090dac650" alt="g002" width="500" height="300">

### 推理结果：
![g002_ser](https://github.com/sss008800/OCR_FaPiao/assets/16535855/1a21a6fd-70e8-459a-8a27-595f9d5e021e)

误报率已经较低，相信再增加几十张训练素材后准确率会更高。
模型比较大，稍后我会放在网盘上提供下载，免费使用。


**下面介绍部署及训练推理**

## 一、部署指导

### 1.安装anaconda环境
### 2.安装paddleocr
> conda环境下执行

```
>> paddlepaddle安装
pip install paddlepaddle==2.5.2 -i https://mirror.baidu.com/pypi/simple
或gpu： pip install paddlepaddle-gpu==2.5.2 -i https://mirror.baidu.com/pypi/simple
>> paddleocr安装
pip install paddleocr==2.7.0.3 -i https://mirror.baidu.com/pypi/simple
pip install PPOCRLabel    -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Polygon3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install lanms-nova  -i https://pypi.tuna.tsinghua.edu.cn/simple

# git官方的PaddleOCR项目，安装需要的依赖
git clone https://gitee.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
# 安装PaddleOCR的依赖
pip install -r requirements.txt
# 安装关键信息抽取任务的依赖
pip install -r ./ppstructure/kie/requirements.txt
```

## 二、训练（可以跳过）

### 1.标注使用PaddleOCR

标注后的txt文件需要转为json，可以使用自己写的transSERjson.py脚本

命令：
```
python tools/train.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh_udml.yml
```

## 三、评估（可以跳过）

```
python tools/eval.py -c ./fapiao/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints="fapiao/ser_vi_layoutxlm_fapiao_trained/best_accuracy/"
```

## 四、推理

```
python tools/infer_kie_token_ser_dxl.py  -c fapiao/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints="./output/ser_vi_layoutxlm_xfund_zh_udml/best_accuracy/"     Global.infer_img=./01ocr/test/  Global.infer_mode=True
```
