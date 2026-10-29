# 用 SwanLab 查看训练过程

训练过程中，我们需要同时观察损失是否下降、验证效果是否改善，以及不同模型和种子的差别。仓库中的训练入口现在把这些信息写入同一套 SwanLab 监测系统。

## 1. 哪些训练入口已经接入

| 训练入口 | 看板中的实验分组 | 主要指标 |
|---|---|---|
| `python -m src.train` | `paper_baselines` | 训练损失、NLL、KL、BCE、验证 MSE |
| `python -m src.comparison` | `classification_comparison` | 训练损失、验证 AUROC/AP、最终分类指标 |
| `python -m src.adni_train` | `adni` | 训练损失、KL、验证 MSE/MAE、模态和时间跨度指标 |
| `python -m paper_odeae.train` | `odeae_reproduction` | MSE、类别交叉熵、KL、L2，以及插值/预测/整体误差 |
| `legacy/train.py` 的 `ImprovedTrainer.train()` | `legacy` | 重构、KL、分类损失、AUROC、敏感度 |

运行批量脚本时，每个模型和种子会建立独立实验。脚本会继承下文的环境变量，所以使用 `paper_odeae.run`、ADNI 批量脚本或容量比较脚本时，不需要另加监测参数。

原始参考快照保留原样；当前维护的训练入口统一调用 [`src/tracking.py`](../src/tracking.py)。

## 2. 先用本地看板查看

已有的 `ode` 环境已安装 SwanLab 和本地看板组件。新环境可通过项目依赖安装：

```bash
conda activate ode
pip install -r requirements.txt
```

默认使用 `local` 模式。训练指标会保存在仓库根目录的 `swanlog/`，不需要登录。

打开一个终端，从仓库根目录启动看板：

```bash
conda activate ode
swanlab watch swanlog --host 127.0.0.1 --port 5092
```

然后访问 `http://127.0.0.1:5092`。如果通过远程 IDE 工作，需要把服务器的 5092 端口转发到本机。

监测接入时已启动这个服务。若端口上已有看板，可以直接打开页面。

## 3. 再启动训练并阅读曲线

以论文复现目录为例，原来的四 GPU 命令可以直接使用：

```bash
python -m paper_odeae.run --gpus 0 1 2 3
```

每个实验保存配置、训练种子、物理 GPU 编号、模型张量设备和参数量。进入单个实验后，可以按下面的顺序观察：

1. 查看 `batch/loss`，确认每批次损失的变化。默认记录第一步和每 20 次参数更新。
2. 查看 `train/loss` 与 `validation/...`。它们每轮记录一次，更适合判断训练趋势和验证效果。
3. 查看 `train/learning_rate`、`batch/gradient_norm` 和系统资源曲线，联系学习率和计算状态理解训练过程。
4. 训练结束后查看 `final/...`，其中保存选定检查点的验证与测试结果。

横轴默认是参数更新步数，`epoch` 指标另行记录训练轮数。SwanLab 按指标名称组织曲线，例如论文复现的未来预测误差位于 `validation/forecast/reconstruction_error`。

每个输出目录还保存 `swanlab.json`，记录运行 ID、日志目录、设备和结束状态。成功、异常和手动中断会分别结束监测实验。

监测发送的是配置和汇总标量。患者逐条记录、预测数组和模型权重继续保存在原来的本地文件中。终端输出仍由现有 `train.log` 等文件保存。

## 4. 怎样调整记录方式

所有入口使用同一组环境变量：

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `SWANLAB_MODE` | `local` | 选择本地、离线、云端或关闭监测 |
| `SWANLAB_PROJECT` | `LatentODE` | 指定项目名称 |
| `SWANLAB_LOG_DIR` | 仓库根目录的 `swanlog/` | 更换日志目录 |
| `SWANLAB_LOG_EVERY` | `20` | 每多少次参数更新记录一条批次指标 |
| `SWANLAB_EXPERIMENT_NAME` | 按分组、种子和输出目录生成 | 手动指定单次实验名称 |
| `SWANLAB_WORKSPACE` | SDK 默认工作区 | 云端模式下选择工作区 |

例如，想给一组实验换名称并提高批次记录频率，可以先设置：

```bash
export SWANLAB_PROJECT=ADNI-ODEAE
export SWANLAB_LOG_EVERY=10
python -m paper_odeae.run --gpus 0 1 2 3
```

项目名称使用普通字符串即可。监测封装已经处理 SwanLab 0.10 对项目环境变量格式的兼容问题。本地看板的多进程回调也会串行写入，避免多个训练任务同时创建图表时发生冲突；模型训练本身继续并行。

## 5. 如果需要云端看板

先用自己的账号登录，再将模式改为 `online`，然后运行原来的训练命令：

```bash
swanlab login
export SWANLAB_MODE=online
export SWANLAB_PROJECT=ADNI-ODEAE
python -m paper_odeae.run --gpus 0 1 2 3
```

云端模式下，输出目录的 `swanlab.json` 会保存实验链接。登录凭据通过 SDK 管理，不写入训练配置。

四种模式的含义如下：

| 模式 | 记录位置和使用方式 |
|---|---|
| `local` | 本地记录，并生成供 `swanlab watch` 使用的看板数据 |
| `offline` | 本地保存 SDK 日志，用于后续同步 |
| `online` | 记录到指定云端项目 |
| `disabled` | 显式关闭监测 |

这些模式与本地看板命令的说明可查阅 [SwanLab 初始化文档](https://docs.swanlab.cn/api/py-init.html)和[离线看板文档](https://docs.swanlab.cn/guide_cloud/offline-board.html)。

## 6. 已经启动的任务怎样接入

运行中的 Python 进程已经加载了启动时的代码，因此修改训练文件不会自动给这些进程增加监测。

为此，仓库提供 [`src/track_history.py`](../src/track_history.py)。它只读取现有 `config.json`、`hardware.json`、`history.json` 和最终 `metrics.json`，将历史轮次补录到 SwanLab，再持续观察后续轮次。它不加载检查点，也不改变优化器或训练进程。

监测接入时发现四个已启动的论文训练任务，已经为它们建立历史跟踪。对应关系保存在 [`attached_runs.json`](../results/swanlab_monitoring/attached_runs.json)。当前任务完成后，批量队列新启动的任务会使用训练代码内的直接监测。

如需手动跟踪另一个既有任务，可指定其输出目录和进程号：

```bash
python -m src.track_history \
  --run paper_odeae/results/local_available/ode_ae/seed_0 \
  --pid 训练进程号
```

只有尚未存在 `swanlab.json` 的目录才允许接入，以避免同一次运行被重复记录。跟踪器会补录已有轮次，并在训练完成时导入最终指标。若训练进程异常结束而没有生成完成指标，监测状态会标记为异常。

## 7. 本次做了哪些验证

监测生命周期测试覆盖了指标展开、记录间隔、正常结束、异常退出、关闭监测和连续运行。还进行了四 GPU 集成检查，实际运行五类训练入口，共完成 10 次小规模训练。

集成检查结果见 [`integration_checks.json`](../results/swanlab_monitoring/integration_checks.json)。检查同时确认了实际模型设备为 CUDA，并检查本地图表文件已经写出。

需要重新验证时，可以运行：

```bash
python -m unittest tests.test_tracking -v
python tools/check_swanlab.py
```

第二条命令会执行短小的 CUDA 训练检查。其数据、检查点和日志独立保存在 `results/swanlab_monitoring/checks/`，不覆盖正式实验结果。
