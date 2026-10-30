# 模型侧最小客户端

此目录仅含公开数据加载器和公开干预规则，无隐藏生成器、动力学方程、配置、患者参数或 oracle。与待评测 gamma 的 `public/` 一同复制/挂载到模型环境；不要将整个项目、synthetic_cf_level3、manifest、config 或 diagnostics 提供给模型。

```bash
# 两个路径均为模型容器内路径；客户端依赖 numpy、pandas、pyarrow
export PYTHONPATH=/mounted/model_client
```

```python
from causalmed_client import PublicDataset, decide
train = PublicDataset('/mounted/public', 'train')
test = PublicDataset('/mounted/public', 'test')
query = test.queries()
```

客户端文件为 synthetic_cf_level3 中安全模块的独立副本，不导入完整生成器包。需要修改客户端时，请同步这两个公开模块；manifest.json 记录交付版本 SHA256。
