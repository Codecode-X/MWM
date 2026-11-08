"""Summarize full mixed-field baseline runs without executing any model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        # Active training may be replacing a history file during this read.
        return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='paper_odeae/results/baselines_full')
    args=parser.parse_args();root=Path(args.root)
    launch=read_json(root/'launch.json') or {}
    rows=[]
    paths=[(f'reference/{m}',ROOT/f'paper_odeae/results/local_available/{m}/seed_0')
                       for m in ['ode_ae','deterministic','lstm']]+[
                       (f'{v}/{m}',root/v/m/'seed_0') for v in ['native','small'] for m in ['latent_ode','ode_rnn']]
    capacity=ROOT/'paper_odeae/results/capacity_matched/deterministic_22545/seed_0'
    if root.resolve()==ROOT/'paper_odeae/results/baselines_full' and capacity.exists():
        paths.append(('capacity/deterministic_22545',capacity))
        for name,folder in [('forward_age','forward_age_from22545'),('forward_zero','forward_from22545')]:
            path=ROOT/f'paper_odeae/results/direction_ablation/{folder}/seed_0'
            if path.exists():paths.append((f'ablation/{name}',path))
    for label,path in paths:
        complete=read_json(path/'metrics.json');history=read_json(path/'history.json')
        entry={'name':label,'path':str(path),'status':'pending'}
        queued=read_json(path/'queue.json')
        if queued:
            entry.update(status=queued['status'],trainable_parameters=queued['target_parameters'],
                         total_parameters=queued['target_parameters'])
        hw=read_json(path/'hardware.json')
        if hw:
            entry.update(total_parameters=hw['parameters'],trainable_parameters=hw.get('trainable_parameters',hw['parameters']))
        if history:
            h=history;entry.update(status='incomplete',epochs=len(h),
                best_validation_overall=min(r['validation']['overall']['reconstruction_error'] for r in h))
        tracking=read_json(path/'swanlab.json')
        if tracking and tracking['status']=='running':
            entry['status']='running'
        elif tracking and tracking['status'] in ['crashed','aborted','tracking_error']:
            entry['status']=tracking['status']
        if complete:
            metrics=complete;hw=json.loads((path/'hardware.json').read_text())
            entry.update(status='complete',epochs=metrics['epochs_run'],best_epoch=metrics['best_epoch'],
                total_parameters=hw['parameters'],
                trainable_parameters=hw.get('trainable_parameters',hw['parameters']),
                test={k:v['reconstruction_error'] for k,v in metrics['test'].items()},
                validation=metrics['validation']['overall']['reconstruction_error'],
                physical_gpu=hw['physical_gpu'],checkpoint_reload_difference=metrics['checkpoint_reload_difference'])
        rows.append(entry)
    lines=['# Latent ODE / ODE-RNN 在 ODE-AE 任务上的比较','',
           '数据使用相同的 local_available 缓存：40 字段，训练/验证/测试患者为 2093/393/414。各患者枚举所有历史截断，使用相同缺失掩码、字段损失和区域计数。', '',
           '所有结果为 seed 0；参考模型取已有完整训练结果。新模型沿用最多 300 轮、至少 50 轮、耐心值 40 的预算。native 使用原 ADNI 模型的 32 维状态与 64 宽网络；small 使用 6 维状态与 16 宽网络。两组都保留原网络的两层 ODE 隐藏层，因此 small 表示维度/宽度对照，并非参数量严格匹配。','',
           '例如，患者有第 1–4 次随访，模型只读取第 1–2 次作为历史：测试插值评价第 1–2 次，测试预测评价第 3–4 次，测试整体评价第 1–4 次。所有指标都在未参加训练的测试患者上计算，并汇总全部历史截断。这里的“插值”指历史区间已有检查时间点的误差，不是额外生成中间时间点的真值。','',
           '每个区域内，各数值字段计算有效观测上的平均平方误差，各类别字段计算平均交叉熵，最后按字段求和。整体误差重新使用全部有效观测计算；因为字段和区域的观测数量不同，它不是插值误差与预测误差的简单算术平均。','',
           '可训练参数量表示本任务中参与优化的参数；总参数量另外包含冻结且未使用的二分类头。三个参考模型没有此闲置头，两个计数相同。','',
           '| 模型/容量组 | 状态 | 可训练参数量 | 总参数量 | 轮数 | 最佳验证整体误差 | 测试插值 | 测试预测 | 测试整体 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    if launch.get('smoke'):
        lines[2:2]=['本报告的新模型是单轮 CUDA 小规模检查，只用于验证运行链路；不能与完整参考任务比较效果。','']
    for r in rows:
        score=r.get('validation',r.get('best_validation_overall'))
        cells=[f'{score:.4f}' if score is not None else '—']
        cells += [f"{r['test'][k]:.4f}" if 'test' in r else '—' for k in ['interpolation','forecast','overall']]
        trainable=f"{r['trainable_parameters']:,}" if 'trainable_parameters' in r else '—'
        total=f"{r['total_parameters']:,}" if 'total_parameters' in r else '—'
        lines.append(f"| {r['name']} | {r['status']} | {trainable} | {total} | {r.get('epochs','—')} | "+' | '.join(cells)+' |')
    by_name={r['name']:r for r in rows}
    if not launch.get('smoke') and all(r['status']=='complete' for r in rows[:7]):
        reference=by_name['reference/ode_ae']['test']['forecast']
        native_rnn=by_name['native/ode_rnn']['test']['forecast']
        native_latent=by_name['native/latent_ode']['test']['forecast']
        deterministic=by_name['reference/deterministic']['test']['forecast']
        lines += ['', '## 完整结果的比较', '',
            '先比较未来预测误差，因为各模型在这个区域都无法读取未来真实检查。', '',
            f'原容量 ODE-RNN 的未来预测误差为 {native_rnn:.4f}，比本地 ODE-AE 低 {100*(1-native_rnn/reference):.1f}%，比确定性版本低 {100*(1-native_rnn/deterministic):.1f}%。原容量 Latent ODE 为 {native_latent:.4f}，比本地 ODE-AE 低 {100*(1-native_latent/reference):.1f}%。', '',
            '小容量 ODE-RNN 的预测误差为 '+f"{by_name['small/ode_rnn']['test']['forecast']:.4f}"+
            '，说明 ODE-RNN 相对当前概率模型的优势在较小网络中仍存在。两个容量的 Latent ODE 结果接近，当前结果没有显示单纯增大容量能明显改善其预测。', '',
            '确定性 ODE-AE 的整体误差仍最低；其历史重构与因果 ODE-RNN 的历史预测并不具有相同信息条件。以上是固定划分、seed 0 的点估计，尚未进行多种子或患者重抽样检验，不能表述为统计显著优势。', '',
            '| 新模型/容量组 | 训练轮数 | 最佳轮次 | 可训练参数量 | 检查点重载差异 |',
            '|---|---:|---:|---:|---:|']
        for r in rows[3:7]:
            lines.append(f"| {r['name']} | {r['epochs']} | {r['best_epoch']} | {r['trainable_parameters']} | {r['checkpoint_reload_difference']:.1g} |")
    if len(rows)>7:
        lines += ['', '## 严格参数量对照', '',
            '新增确定性 ODE-AE 固定潜在维度 6，只将 INIT/UPDATE/DECODER 隐藏层宽度改为 154、前后向 ODE 隐藏层宽度改为 118，全部参数均参与训练，总计恰为 22,545，与原容量 ODE-RNN 的可训练参数量一致。', '',
            '该任务用四张 RTX 4090 进行 DDP，保持全局 batch size 32，每卡 8 个样本。重构损失按全局字段观测数量归一化，随后补偿 DDP 的梯度平均；不是对每卡局部平均误差直接求平均。数据、评分和训练预算沿用原设置。分卡数据增强与自适应 ODE 步长可能造成数值轨迹差异，参数量匹配也不消除历史评分信息条件的差别。','']
        matched=rows[7]
        if matched['status']=='complete':
            comparator=by_name['native/ode_rnn']
            change=100*(1-matched['test']['forecast']/comparator['test']['forecast'])
            lines += [f"22,545 参数确定性 ODE-AE 的测试预测误差为 {matched['test']['forecast']:.4f}，相对同参数量 ODE-RNN 的误差降低比例为 {change:.2f}%（负值表示误差更高）。",'']
    if any(r['name'].startswith('ablation/') for r in rows):
        lines += ['', '## 正向历史编码与初始化对照', '',
            '两个新增确定性版本都按时间正向读取历史，并把最终编码结果解释为历史截止点处的状态；从该状态独立向过去积分重建历史、向未来积分预测。二者保留同样的 MLP 观测更新与 ODE 生成器，不改成 GRU，也不增加未来预测反馈。', '',
            'forward_age 保留原截止点年龄 INIT 网络，共 22,545 参数；forward_zero 用全零状态代替 INIT，删除该网络后共 21,307 参数。共享网络宽度均为 MLP 154、ODE 118，潜在维度均为 6。没有加入闲置参数来凑齐计数。两个版本相互比较用于检查初始化方式；与原反向版本比较时，还同时涉及编码方向和生成锚点变化。', '',
            '这两个实验排在当前四卡容量对照之后，每次使用四卡 DDP、全局 batch 32、seed 0 和相同完整训练预算。queued 表示尚未开始，checking 表示正在做四卡一致性及小规模检查；测试列只在正式任务结束后填写。','']
    lines += ['', 'Latent ODE 保留原 ADNI 适配器的正向历史编码及截止点后验，通过生成 ODE 向过去和未来积分以覆盖全部评分时间。ODE-RNN 保留因果自回归方式，每个历史时间点先预测再读取该时刻观测；因此它的历史插值项与利用整个历史重构过去的模型有信息条件上的差别，未来预测项应作为主要共同对照。', '',
              '类别输入使用独热编码，未来反馈使用预测类别概率；评分使用类别 logits 的交叉熵。旧 58 维检查点没有兼容的字段及类别输出，本次新训练混合字段输出层。', '',
              '评价都使用相同 evaluate 函数；概率模型以潜在均值给出点预测。训练损失同为字段 MSE＋CCE＋1×KL＋0.02×ODE L2，未采用原 ADNI 实验的弱 KL 设置。任务变化后的结果用于本地共同协议比较，仍不能直接与论文绝对数值比较。','']
    lines += ['原模型保留此前以十年为单位的相对时间参数化，ODE-AE 使用年。实际检查时间和评估目标相同，但时间单位影响向量场参数与 L2 的关系，因此这里的训练先验并不完全等价。','']
    root.mkdir(parents=True,exist_ok=True)
    import os
    def atomic_write(name,value):
        temporary=root/f'{name}.{os.getpid()}.tmp';temporary.write_text(value);temporary.replace(root/name)
    atomic_write('summary.md','\n'.join(lines))
    atomic_write('summary.json',json.dumps({'updated_at':datetime.now(timezone.utc).isoformat(),
        'smoke':launch.get('smoke',False),'dataset_sha256':launch.get('dataset_sha256'),'runs':rows},indent=2)+'\n')
    print(root/'summary.md')


if __name__=='__main__':main()
