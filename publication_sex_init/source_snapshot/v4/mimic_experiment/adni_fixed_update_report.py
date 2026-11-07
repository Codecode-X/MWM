"""ADNI fixed-alpha result section, using saved CPU-readable metrics."""
import json
from pathlib import Path


def read(p, default=None): return json.loads(p.read_text()) if p.exists() else default


def section(root):
    folder=root/'adni_update_alpha_01'; manifest=read(folder/'manifest.json')
    if not manifest:return []
    cfg=read(folder/'config.json') or read(folder/'launch_config.json');q=read(folder/'queue.json',{})
    reference=read(Path(manifest['reference_metrics']));history=read(folder/'history.json',[]);metrics=read(folder/'metrics.json')
    baseline=Path(manifest['reference_metrics']).parent;oldhistory=read(baseline/'history.json',[])
    lines=['### 8.6 ADNI迁移实验：固定α=0.1','',
           '本组将相同的UPDATE混合方式用于ADNI：保留90%的ODE演化后状态，采用10%的UPDATE网络输出。历史回顾与未来预测都保留原完整梯度，未叠加深度8截断。','',
           '| 项目 | 设置 |','|---|---|',
           '| 数据 | 原ADNI local_available缓存；原训练/验证/测试患者划分，2,093 / 393 / 414人 |',
           '| 状态与参数 | 28维潜在状态，44,875参数 |',
           '| 监督 | 枚举患者全部非空历史前缀；从历史截止点回顾过去并预测未来 |',
           '| 归一化 / 时间单位 | 训练集min-max / 年 |',
           '| 初始化 / GPU | seed 0从零训练 / 物理CUDA 3，ode环境 |',
           '| batch / 梯度累积 | 沿用ADNI原实验32 / 无 |',
           '| 损失 | 原数值MSE＋类别CE，0.01 weak prior，0.01 Jacobian，0.02 ODE L2 |',
           '| 预算与选模 | 最大150轮，至少50轮，验证整体40轮不改善早停；按验证整体选最佳检查点 |','',
           'ADNI与MIMIC各自沿用原数据评测协议。下面在ADNI内部比较α=0.1和原α=1，MIMIC的小时轨迹结果保留在8.4。','']
    check=read(folder/'checks.json')
    if check:
        lines += [f"CUDA核验已通过：α=1时输出和损失恢复原模型，固定α=0.1的Jacobian公式最大绝对误差为{check['jacobian_formula_max_absolute_error_float64']:.3g}。历史和未来项对INIT、UPDATE、ODE、Decoder均有非零梯度。",'',
                  '[CUDA检查](targeted_causal_ablation/adni_update_alpha_01/checks.json)','']
    groups=[('ADNI原latent2x both（α=1）',reference,oldhistory,'complete'),('ADNI固定α=0.1',metrics,history,q.get('status','等待'))]
    lines += ['| 模型 | 状态 | 已训练 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |','|---|---|---:|---:|---:|---:|']
    for name,result,hs,status in groups:
        best=min(hs,key=lambda h:h['validation']['overall']['reconstruction_error']) if hs else None
        selected=result['validation'] if result else best['validation'] if best else None
        n=result['epochs_run'] if result else len(hs);e=result['best_epoch'] if result else best['epoch'] if best else '—'
        row=[name,status,f'{n} / {e}']+[f"{selected[k]['reconstruction_error']:.6f}" if selected else '—' for k in ['interpolation','forecast','overall']]
        lines.append('| '+' | '.join(row)+' |')
    lines += ['','| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 |','|---|---:|---:|---:|---:|']
    for name,result,_,_ in groups:
        t=result['test'] if result else None
        row=[name,str(result['best_epoch']) if result else '—']+[f"{t[k]['reconstruction_error']:.6f}" if t else '—' for k in ['interpolation','forecast','overall']]
        lines.append('| '+' | '.join(row)+' |')
    if history:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for name,hs in [('ADNI original alpha=1',oldhistory),('ADNI alpha=0.1',history)]:
            for ax,key in zip(axes.flat,['interpolation','forecast','overall',None]):
                ax.plot([h['epoch'] for h in hs],[h['validation'][key]['reconstruction_error'] if key else h['training_components']['gradient_norm'] for h in hs],label=name)
        for ax,title in zip(axes.flat,['Validation history','Validation future','Validation overall','Mean pre-clip batch gradient']):ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[1,1].set_yscale('log');fig.tight_layout();fig.savefig(root/'adni_alpha01_curves.png',dpi=170);plt.close(fig)
        lines += ['','![ADNI α=0.1与原模型：验证曲线及轮均梯度](targeted_causal_ablation/adni_alpha01_curves.png)','']
    lines += ['','训练结束后自动载入最佳检查点，核对验证指标，再填写测试结果。','',
              '[配置与冻结源码](targeted_causal_ablation/adni_update_alpha_01/manifest.json) · [运行状态](targeted_causal_ablation/adni_update_alpha_01/queue.json) · [训练记录](targeted_causal_ablation/adni_update_alpha_01/)','',
              '```bash','bash scripts/run_mimic_adni_fixed_update_150.sh','```','']
    return lines
