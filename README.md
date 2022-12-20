# CO2NET
Deep learning for characterizing CO2 migration in time-lapse seismic images

![cc](https://www.sciencedirect.com/science/article/abs/pii/S0016236122036304) [github标签网站](https://github.com/shenghanlin/CO2NET)

项目描述从这里开始。

## Introduction

这些说明将为您提供在本地计算机上启动和运行的项目副本，以进行开发和测试。有关如何在实时系统上部署项目的说明，请参阅部署。

<!-- more -->

### Preparation

```c++
python.__version__  '3.6.2'
tf.__version__      '1.13.1'
keras.__version__   '2.3.1'
```


## Data

I compress the data and then reduce the number of validation datasets because validation datasets do not affect the training. I have uploaded all these datasets to GDrive.

https://drive.google.com/drive/folders/1hHDkq3qyqNUU3V221OaWHGTcn458YyNg?usp=sharing

### Training


```python
python train.py
```

### 带有代码风格的测试

解释这些测试的测试内容和原因。

```c++
Give an example
```

## 部署

对以上的安装步骤进行补充说明，描述如何在在线系统上部署该项目的其他说明。

## 依赖

- [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - 使用的Web框架
- [Maven](https://maven.apache.org/) - 依赖管理
- [ROME](https://rometools.github.io/rome/) - 用于生成RSS源

## 贡献流程

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) ，了解我们的行为准则以及向我们提交拉取请求的流程。

## 版本控制

我们使用 [SemVer](http://semver.org/) 进行版本控制。 对于可用的版本， 请参阅 [此存储库中的tags](https://github.com/your/project/tags).

## 作者

- **Ah Dai** - *初步工作* - [ahdaidawn](https://github.com/AhdaiDawn)

另请参阅参与此项目的[贡献者](https://github.com/your/project/contributors)列表。

## 版权说明

署名-非商业性使用-相同方式共享 4.0 国际 (CC BY-NC-SA 4.0)

![fd](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)
该项目签署了MIT授权许可 - 有关详细信息，请参阅 [LICENSE.md](LICENSE.md) 文件。

## FAQ

当然了，如果你不想回答一些非常重复的问题，我想你需要一份 FAQ 来记录一些常见问题。

## 致谢

- 给任何使用此项目的人的提示
- 线路图
- 等等
