"""Report public-fact training and separately scored counterfactual submissions."""
import json
from pathlib import Path


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def generate(output):
    out=Path(output);queue=read(out/'queue.json',{});manifest=read(out/'manifest.json',{});rows=[]
    order=manifest.get('order',['gamma_1']);is_odeae=manifest.get('model_name')=='ODE-AE'
    title='ODE-AE' if is_odeae else 'latent2x both';gpu=manifest.get('physical_gpu',3)
    structure=('ODE-AE保留6维随机状态、12维反向编码状态、16宽网络及独立编码/生成ODE。INIT接收标准化静态c；编码UPDATE吸收历史观测和动作，生成阶段使用6维治疗校正模块。总参数1,843。' if is_odeae else
               '保留latent2x both的28维状态、159宽INIT/UPDATE/DECODER及188宽共享ODE。INIT接收标准化静态c；UPDATE接收观测、掩码、动作独热编码及事件类型。治疗切换采用新旧动作条件UPDATE输出之差形成残差位移，保持原治疗时位移为零。参数量为31,314。')
    loss=('目标为三字段标准化MSE＋1×Gaussian后验KL＋0.02×编码与生成ODE参数L2。' if is_odeae else
          '目标为三字段标准化MSE＋0.02×ODE参数L2＋0.01×weak prior＋0.01×完整Jacobian平方范数。')
    lines=[f'# CausalMed：{title} 的干预与反事实推演实验','',
           '## 1. 任务如何进行','',
           '每组使用10,000位训练患者、2,000位验证患者、2,000位测试患者。模型学习公开事实轨迹中的Y、X1、X2、静态c和治疗区间。测试在首次达到3年的检查后分叉，从同一个患者状态分别执行五种公开策略，推演至8年边界。','',
           '动态策略在分叉时使用已经观察到的X2，之后使用各自分支预测的X2。未来检查只提供决策时刻，状态通过共享ODE演化，并在治疗切换时接受动作校正。','',
           '## 2. 模型与训练设置','',
           structure,'',
           loss+'归一化仅拟合训练事实记录。验证与训练使用记录中的事实治疗路径，未来阶段只执行动作校正；验证整体误差选择检查点。','',
           f'本轮仅运行gamma=1。全部模型计算使用物理GPU {gpu}、ode环境，seed为0，batch为128，最多300轮、至少50轮、早停耐心40轮。训练接入SwanLab。','',
           '## 3. 当前进度与事实验证','',
           '| gamma | 阶段 | 已训练 / 最佳轮次 | 验证整体标准化误差 | 验证未来Y RMSE |','|---|---|---:|---:|---:|']
    for key in order:
        folder=out/key;history=read(folder/'model/history.json',[]);completed=read(folder/'model/training_complete.json')
        best=min(history,key=lambda h:h['validation']['overall']['reconstruction_error']) if history else None
        validation=completed['validation'] if completed else best['validation'] if best else None
        state=queue.get('jobs',{}).get(key,{});status=state.get('status','waiting')+' / '+state.get('phase','waiting')
        epoch=completed['best_epoch'] if completed else best['epoch'] if best else '—'
        lines.append(f"| {key[6:]} | {status} | {len(history)} / {epoch} | {validation['overall']['reconstruction_error']:.5f} | {validation['future_raw_rmse'][0]:.5f} |" if validation else f"| {key[6:]} | {status} | 0 / — | — | — |")
        rows.append({'key':key,'history':history,'completed':completed,'action':read(folder/'evaluation/with_actions/metrics.json'),
                     'off':read(folder/'evaluation/action_off/metrics.json')})
    lines+=['','## 4. 反事实轨迹误差','',
            'CF-RMSE比较预测Y与配对的无噪声反事实Y。全轨迹覆盖公开逐月查询点，1/2/3年栏使用精确终点。动作关闭对照复用同一检查点和历史状态，只关闭分叉后的动作校正。','',
            '| gamma | 版本 | 全轨迹CF-RMSE | 1年CF-RMSE | 2年CF-RMSE | 3年CF-RMSE |','|---|---|---:|---:|---:|---:|']
    for row in rows:
        for name,key in [('动作输入','action'),('关闭未来动作','off')]:
            m=row[key]
            values=[row['key'][6:],name]+([f"{m['cf_rmse_trajectory']:.5f}"]+[f"{m['horizons'][str(t)]['cf_rmse']:.5f}" for t in [1.,2.,3.]] if m else ['—']*4)
            lines.append('| '+' | '.join(values)+' |')
    lines+=['','## 5. 相对Never Treat的个体效应与策略选择','',
            'sqrt(PEHE)评价个体治疗效应；ATE Error评价人群平均效应。策略效用使用公开固定成本：Y减去3×治疗年数。预测治疗年数由模型自己的动作区间计算。','',
            '| gamma | 年限 | Always相对Never的sqrt(PEHE) | ATE Error | 真实ATE | 预测ATE | 策略Regret | 最优策略准确率 |','|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if not row['action']:continue
        for t,h in row['action']['horizons'].items():
            effect=h['effects_vs_never']['1']
            lines.append(f"| {row['key'][6:]} | {t} | {effect['sqrt_pehe']:.5f} | {effect['ate_error']:.5f} | {effect['true_ate']:.5f} | {effect['predicted_ate']:.5f} | {h['policy_regret']:.5f} | {100*h['optimal_policy_accuracy']:.2f}% |")
    lines+=['','### 各策略的3年CF-RMSE','',
            '| gamma | Never | Always | Delayed | Aggressive | Conservative |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['action'];h=m['horizons']['3.0'] if m else None
        lines.append('| '+' | '.join([row['key'][6:]]+[f"{h['by_policy_rmse'][str(p)]:.5f}" if h else '—' for p in range(5)])+' |')
    lines+=['','## 6. 数据与执行记录','',
            '训练和预测在独立文件系统命名空间中运行，只挂载公开数据、模型源码、当前输出和SwanLab日志。生成器、隐藏参数和反事实真值不出现在模型命名空间。模型先写完预测与自有治疗区间，固定文件SHA256，再由独立的CPU统计评分进程读取真值。','',
            '[方法与实验说明](../../latent2x_experiment/README.zh-CN.md) · [运行队列](queue.json) · [固定配置与源码](manifest.json)','',
            '每组model/中保存best.pt、last.pt、history.json、hardware.json、isolation.json及两组预测。evaluation/保存官方评分器输出。报告每30秒更新。','',
            '```bash','bash scripts/run_causalmed_latent2x_both.sh','```','']
    (out/'summary.md').write_text('\n'.join(lines))
    (out/'comparison.json').write_text(json.dumps({'datasets':[{k:v for k,v in row.items() if k!='history'} for row in rows]},indent=2)+'\n')
