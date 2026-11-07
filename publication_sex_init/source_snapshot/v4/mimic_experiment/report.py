"""Readable data audit and current/final MIMIC experiment report; CPU statistics."""
import json
from pathlib import Path


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def generate(out,root):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);root=Path(root)
    data=read(root/'dataset_summary.json',{});queue=read(out/'queue.json',{});run=out/'seed_0';history=read(run/'history.json',[]);final=read(run/'metrics.json')
    lines=['# MIMIC-IV：latent2x both 的 ICU 纵向预测实验','',
           '## 1. 从已有数据建立什么任务','',
           '本次使用本地 MIMIC-IV 3.1。每位患者取首次 ICU 住院，要求 ICU 时长至少48小时，且前48小时完整落在住院期间、没有早于窗口终点的死亡记录。再要求前24小时与后24小时各至少有6个带观测的小时窗口。患者按稳定哈希分为训练、验证和测试，比例约70%/15%/15%。','',
           '主要任务是：读取前24小时可获得的观测，重构历史并预测之后24小时的生理与GCS观测。训练时每位患者每轮选一个6、12、18或24小时历史截止点；验证选点固定使用24小时。','',
           '## 2. 数据整理到哪一步','',f"当前阶段：**{queue.get('status','preparing')}**。",'']
    if data:
        lines+=['| 项目 | 数量 |','|---|---:|',f"| 符合完整观测条件的患者 | {data['patients']:,} |"]
        for key in ['train','validation','test']:lines.append(f"| {key}患者 | {data['patient_counts'][key]:,} |")
        lines += [f"| 原始字段 | {data['features']} |",f"| 类别展开后维度 | {data['expanded_features']} |",'',
                  '可用时间取测量时间与入库时间的较晚者。数值在同一可用时间小时内取均值，GCS取众数；空小时被移除，其余缺失字段保留掩码。性别只在第一条真实观测记录中提供。归一化仅由训练患者的有效观测拟合，验证和测试不裁剪到训练范围。','',
                  f"完整数据与来源映射位于 `{root}`，训练缓存为 `dataset.pkl`。",'']
    else:
        for table in ['chartevents','labevents']:
            progress=read(root/(table+'_audit.json'),read(root/(table+'_progress.json'),{}))
            if progress:lines.append(f"{table} 已扫描 {progress.get('source_rows',0):,} 行，保留窗口内有效值 {progress.get('retained_event_values',0):,} 条。")
    lines+=['','## 3. 模型与训练方式','',
            '模型保留28维潜在状态、159宽的INIT/UPDATE/DECODER、188宽的共享ODE，以及原自适应求解器。输入输出层适配本数据字段。ODE以小时推进；INIT接收由anchor_age和anchor_year推算的入ICU年龄，再换算到历史截止时刻，单位为年、输入时除以100。','',
            '损失为原字段MSE＋类别交叉熵＋0.02×ODE参数L2＋0.01×观测校正前误差＋0.01×Jacobian平方范数。Jacobian仍对完整28×28导数的784项平方求和，再对有效生成状态求平均，采样位置停止梯度。','',
            '使用4张RTX 4090做DDP，训练seed 0，全局batch 128，每卡32。Adam学习率0.003，最多300轮，最少50轮，连续40轮验证整体误差未改善则早停。四卡按全局字段观测数和全局有效状态数归约损失；训练分片不重复、不遗漏患者。全部训练接入SwanLab。','']
    hardware=read(run/'hardware.json',{})
    if hardware:lines.append(f"实际模型参数量：**{hardware['parameters']:,}**。物理GPU：{hardware['physical_gpu']}。")
    check=read(out/'checks/step/distributed_step_check.json')
    if check:lines.append(f"四卡与单卡同一批数据的损失差为 {check['loss_difference']:.3g}，SGD更新后最大参数差为 {check['parameter_step_max_difference']:.3g}。")
    lines+=['','## 4. 原训练目标下的误差','',
            '各数值字段先在训练集min-max尺度上计算MSE，各类别字段计算交叉熵，然后按字段求和。验证和测试整体误差重新汇总全部有效观测。','',
            '| 状态 | 已训练轮次 | 最佳轮次 | 验证整体 | 验证未来 | 测试历史 | 测试未来 | 测试整体 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    best=min(history,key=lambda r:r['validation']['overall']['reconstruction_error']) if history else None
    def value(item,region):return f"{item[region]['reconstruction_error']:.5f}" if item else '—'
    validation=final['validation'] if final else best['validation'] if best else None
    test=final['test'] if final else None
    lines.append('| '+' | '.join([final['status'] if final else queue.get('status','preparing'),str(final['epochs_run'] if final else len(history)),str(final['best_epoch'] if final else best['epoch'] if best else '—'),
                                  value(validation,'overall'),value(validation,'forecast'),value(test,'interpolation'),value(test,'forecast'),value(test,'overall')])+' |')
    if history:
        avg=sum(r['epoch_seconds'] for r in history)/len(history)
        lines+=['',f"已完成轮次的平均耗时约 {avg/60:.2f} 分钟/轮。"]
    if final:
        lines+=['','## 5. 未来不同时间范围的预测','',
                '下面固定读取前24小时，分别累计评价后续6、12、24小时内的有效观测。','',
                '| 未来累计范围 | MSE＋交叉熵 | 数值MSE之和 | 类别交叉熵之和 | 有效观测数 |','|---|---:|---:|---:|---:|']
        for h in [6,12,24]:
            m=test[f'forecast_{h}h'];lines.append(f"| {h}小时 | {m['reconstruction_error']:.5f} | {m['sum_feature_mse']:.5f} | {m['sum_feature_cce']:.5f} | {m['observed_scalars']:,} |")
        lines+=['','## 6. 转回原始单位，与保持最近观测值比较','',
                '参考方法只保留截止时刻已获得的最近值，之后保持不变。历史里从未测到的数值字段用训练均值，类别字段用训练众数。它与模型使用相同历史和未来观测掩码。','',
                '| 数值字段 | 单位 | 模型MAE | 保持最近值MAE | 模型RMSE | 保持最近值RMSE |','|---|---|---:|---:|---:|---:|']
        for name,m in test['clinical_forecast'].items():
            if 'mae' not in m:continue
            ref=test['carry_forward'][name];lines.append(f"| {name} | {m['unit']} | {m['mae']:.4f} | {ref['mae']:.4f} | {m['rmse']:.4f} | {ref['rmse']:.4f} |")
        lines+=['',f"将每个数值字段的原始单位MSE除以训练标准差平方、再对字段等权平均：模型为 **{test['mean_standardized_numeric_mse']:.5f}**，保持最近值为 **{test['carry_forward_mean_standardized_numeric_mse']:.5f}**。",'',
                '| 类别字段 | 模型准确率 | 保持最近值准确率 |','|---|---:|---:|']
        for name,m in test['clinical_forecast'].items():
            if 'accuracy' in m:lines.append(f"| {name} | {m['accuracy']*100:.2f}% | {test['carry_forward'][name]['accuracy']*100:.2f}% |")
        lines+=['','## 7. 只获得更短历史时的结果','',
                '| 历史截止 | 有效患者 | 未来评价范围 | 未来MSE＋交叉熵 |','|---|---:|---|---:|']
        for h,m in final['test_early_contexts'].items():lines.append(f"| {h}小时 | {m['patients']:,} | 第{h}小时后至第48小时 | {m['forecast']['reconstruction_error']:.5f} |")
    lines+=['','## 8. 记录与复现','',
            '[实验设计](../../README.zh-CN.md) · [配置](config.json) · [队列](queue.json) · [来源与数据核对](data_verification.json)','',
            '模型完成后，完整结果保存在 `seed_0/metrics.json`，逐轮记录在 `seed_0/history.json`。`rank_0_devices.json`至`rank_3_devices.json`分别记录实际张量设备；源码快照固定实际执行版本。','',
            '```bash','bash scripts/run_mimic_latent2x_both.sh','```','']
    (out/'summary.md').write_text('\n'.join(lines))
    if final:training_plot(out,history)


def training_plot(out,history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not history:return
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    x=[r['epoch'] for r in history]
    axes[0].plot(x,[r['validation']['overall']['reconstruction_error'] for r in history],label='Validation overall')
    axes[0].plot(x,[r['validation']['forecast']['reconstruction_error'] for r in history],label='Validation forecast');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Mixed-field error')
    axes[1].plot(x,[r['training_components']['weighted_prior'] for r in history],label='0.01 prior')
    axes[1].plot(x,[r['training_components']['weighted_jacobian'] for r in history],label='0.01 Jacobian');axes[1].legend();axes[1].set(xlabel='Epoch',ylabel='Regularization contribution')
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(Path(out)/('training_curves.'+ext),dpi=180)
    plt.close(fig)
