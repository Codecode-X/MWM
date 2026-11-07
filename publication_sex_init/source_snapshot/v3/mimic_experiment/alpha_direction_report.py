"""Dataset-local forward/reverse alpha=.1 comparison with actual budgets."""
import json
from pathlib import Path


def read(p, default=None): return json.loads(p.read_text()) if p.exists() else default


def section(root):
    study=root/'direction_alpha01';manifest=read(study/'manifest.json')
    if not manifest:return []
    queue=read(study/'queue.json',{});jobs=queue.get('jobs',{})
    lines=['### 8.7 α=0.1门控残差：ADNI与MIMIC的正序、反序对照','',
           '此前的α消融使用正序吸收历史。本轮固定α=0.1，在相同网络、数据和损失下，比较此前定义的正序与反序历史编码策略。ADNI反序在CUDA 2继续，MIMIC反序在CUDA 3独立训练；新增训练不使用梯度累积，最大150轮。','',
           '两种策略每次观测校正都使用同一形式：','',
           r'\[z^+=0.9z^-+0.1U_\phi(z^-,y).\]','',
           '| 计算环节 | 正序组 | 反序组 |','|---|---|---|',
           '| INIT输入 | 最后一条已知历史检查处的年龄 | 相同年龄、相同INIT网络 |',
           '| 放置初始状态的时刻 | 首次历史检查 | 最后一次历史检查 |',
           '| 主编码读取顺序 | 最早 → 最近，ODE向未来演化 | 最近 → 最早，ODE向过去演化 |',
           '| 主编码结束后的状态 | 历史截止点状态 | 首次历史检查状态，已吸收完整历史 |',
           '| 自由生成 | 从截止点分别回顾过去、预测未来 | 从首次检查状态沿ODE向未来生成全部历史和未来 |',
           '| 生成时是否再UPDATE | 否 | 否 |',
           '| weak prior | 正序读取时，在UPDATE前预测下一次观测 | 独立的正序辅助路径；同样α=0.1，仅提供相同weak prior监督 |','',
           '反序主路径和正序辅助路径使用同一INIT、UPDATE、共享ODE和Decoder，没有新增网络参数。辅助路径的最终状态不进入主重建或未来预测。ODE Jacobian正则仍对主生成轨迹采样，采样位置停止梯度；ODE L2仍对共享场计数一次。','',
           '| 共同设置 | ADNI | MIMIC |','|---|---|---|',
           '| 状态维度 / 总参数 | 28 / 44,875 | 28 / 43,280 |',
           '| batch / 梯度累积 | 32 / 无 | 128 / 无 |',
           '| 时间 / 归一化 | 年 / 训练集min-max | 小时 / 原MIMIC训练集min-max |',
           '| 训练历史 | 枚举全部非空历史前缀 | 原6/12/18/24小时截止点采样 |',
           '| 初始化与优化 | seed 0，原Adam和学习率衰减，梯度裁剪10 | 相同 |',
           '| 预算和选模 | 正序和反序最大150轮，按最佳验证整体选模 | 正序复用已有237轮；反序最大150轮，均按最佳验证整体选模 |',
           '| 损失 | 原MSE＋类别CE，0.01 weak prior，0.01 Jacobian，0.02 ODE L2 | 相同权重，保留原字段归一化与评测协议 |',
           '| ODE求解器 | Tsit5，rtol=1e−4、atol=1e−6 | 相同 |','',
           'ADNI正序组复用8.6的α=0.1结果。按当前安排，MIMIC正序直接复用已有237轮训练数据，不再补跑150轮正序对照。原选择规则取验证整体最优的第233轮best.pt作为测试模型；第237轮是该组实际完成的训练轮数。','',
           'ADNI反序保留原进程在CUDA 2运行，未中断或重新初始化。MIMIC反序已改到CUDA 3，与ADNI并行。MIMIC反序完成后，在CUDA 3对已有正序best.pt补齐测试指标，此步骤不执行训练或optimizer.step()。','',
           '#### CUDA核验','',
           '| 训练任务 | 状态 | 参数量 | α的Jacobian公式误差 | weak prior与正序逐位一致 | 未来观测隔离 |','|---|---|---:|---:|---|---|']
    for name in ['adni_reverse','mimic_reverse']:
        check=read(study/name/'checks.json') or read(study/name/'preflight_checks.json')
        vals=[name,check['status'] if check else jobs.get(name,{}).get('status','queued'),str(check['parameters']) if check else '—',
              f"{check['jacobian_formula_max_absolute_error_float64']:.3g}" if check else '—',
              '是' if check and check['causal_prior_bitwise_equal_to_forward_alpha01'] else '—',
              '通过' if check and check['future_observations_do_not_change_outputs'] else '—']
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['','检查还包括：两种方向相同seed下初始张量一致、相同参数量、与已有反序实现的输出逐位一致、只有一次历史检查时两种方向的有效预测一致、负时间间隔用于向过去编码，以及所有网络都有有限梯度。核验不执行优化步骤。','']
    for dataset in ['adni','mimic']:
        reverse=study/(dataset+'_reverse')
        forward=root/'adni_update_alpha_01' if dataset=='adni' else root/'update_alpha_01'
        fwq=read(forward/'queue.json',{}).get('status','queued');rvq=jobs.get(dataset+'_reverse',{}).get('status','queued')
        if dataset=='mimic':fwq='复用既有237轮'
        groups=[('正序α=0.1',forward,fwq),('反序α=0.1',reverse,rvq)]
        title='同预算验证和测试' if dataset=='adni' else '正序237轮与反序最多150轮的结果比较'
        lines += [f'#### {dataset.upper()}：{title}','',
                  '| 模型 | 状态 | 已训练 / 最佳轮次 | 验证历史 | 验证未来 | 验证整体 |','|---|---|---:|---:|---:|---:|']
        data=[]
        for label,folder,status in groups:
            hs=read(folder/'history.json',[]);result=read(folder/'metrics.json')
            if dataset=='mimic' and folder==forward:
                result=result or read(study/'mimic_forward_existing/metrics.json')
            best=min(hs,key=lambda h:h['validation']['overall']['reconstruction_error']) if hs else None
            selected=result['validation'] if result else best['validation'] if best else None
            count=result['epochs_run'] if result else len(hs);epoch=result['best_epoch'] if result else best['epoch'] if best else '—'
            vals=[label,'complete' if result else status,f'{count} / {epoch}']+[f"{selected[k]['reconstruction_error']:.6f}" if selected else '—' for k in ['interpolation','forecast','overall']]
            lines.append('| '+' | '.join(vals)+' |');data.append((label,folder,hs,result))
        lines += ['','| 模型 | 最佳轮次 | 测试历史 | 测试未来 | 测试整体 |','|---|---:|---:|---:|---:|']
        for label,folder,hs,result in data:
            test=result['test'] if result else None
            vals=[label,str(result['best_epoch']) if result else '—']+[f"{test[k]['reconstruction_error']:.6f}" if test else '—' for k in ['interpolation','forecast','overall']]
            lines.append('| '+' | '.join(vals)+' |')
        if all(row[3] for row in data):
            f,r=data[0][3],data[1][3]
            dh=r['test']['interpolation']['reconstruction_error']-f['test']['interpolation']['reconstruction_error']
            df=r['test']['forecast']['reconstruction_error']-f['test']['forecast']['reconstruction_error']
            prefix='同预算下' if dataset=='adni' else '按表中实际训练轮数'
            lines += ['',f'{prefix}，反序减正序的测试历史误差差值为{dh:+.6f}，测试未来差值为{df:+.6f}。','']
        if any(row[2] for row in data):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(2,2,figsize=(12,8))
            for (label,folder,hs,result),name in zip(data,['forward alpha=0.1','reverse alpha=0.1']):
                if not hs:continue
                for ax,key in zip(axes.flat,['interpolation','forecast','overall',None]):
                    ax.plot([h['epoch'] for h in hs],[h['validation'][key]['reconstruction_error'] if key else h['training_components']['gradient_norm'] for h in hs],label=name)
            for ax,title in zip(axes.flat,['Validation history','Validation future','Validation overall','Mean pre-clip batch gradient']):ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
            axes[1,1].set_yscale('log');fig.tight_layout();fig.savefig(study/(dataset+'_direction_curves.png'),dpi=170);plt.close(fig)
            lines += ['',f'![{dataset.upper()}正序/反序α=0.1训练曲线](targeted_causal_ablation/direction_alpha01/{dataset}_direction_curves.png)','']
    lines += ['','[CUDA 2/3队列与进度](targeted_causal_ablation/direction_alpha01/queue.json) · [实验设置](targeted_causal_ablation/direction_alpha01/manifest.json) · [本次安排调整](targeted_causal_ablation/direction_alpha01/execution_revision.json)','',
              '[ADNI反序记录](targeted_causal_ablation/direction_alpha01/adni_reverse/) · [MIMIC反序记录](targeted_causal_ablation/direction_alpha01/mimic_reverse/) · [已有MIMIC正序训练](targeted_causal_ablation/update_alpha_01/) · [正序检查点测试记录](targeted_causal_ablation/direction_alpha01/mimic_forward_existing/)','',
              '每组通过CUDA检查后训练，完成时自动载入最佳检查点测试并回填本节。','',
              '```bash','bash scripts/run_alpha01_direction_adni_mimic.sh','```','']
    return lines
