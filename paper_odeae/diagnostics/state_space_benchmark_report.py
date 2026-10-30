"""CPU-only figure assembly and narrative from completed CUDA diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
from .state_space_benchmark import RUNS,OUT,ROOT,save
from .state_space import COLORS,LABELS


def read_runs(out):
    runs=[]
    for name,title,family,relative in RUNS:
        folder=out/name
        if not (folder/'statistics.json').exists():continue
        runs.append({'name':name,'title':title,'family':family,'folder':folder,
                     'stats':json.loads((folder/'statistics.json').read_text()),
                     'arrays':dict(np.load(folder/'states.npz')),
                     'field':dict(np.load(folder/'projected_field.npz')),
                     'metadata':json.loads((folder/'metadata.json').read_text()),
                     'paths':json.loads((folder/'patient_paths.json').read_text()),
                     'forecasts':json.loads((folder/'forecast_cases.json').read_text()),
                     'velocity_plane':dict(np.load(folder/'velocity_plane.npz')),
                     'velocity_stats':json.loads((folder/'velocity_plane.json').read_text()),
                     'velocity_paths':json.loads((folder/'velocity_plane_paths.json').read_text())})
    if runs:
        for run in runs:assert run['metadata']==runs[0]['metadata']
    return runs


def figures(out,runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize,to_rgb
    for coloring in ['diagnosis','mmse','age']:
        fig,axes=plt.subplots(3,3,figsize=(17,15))
        for ax,run in zip(axes.flat,runs):
            xy=run['arrays']['coordinates'];meta=run['metadata'];test=np.array([r['split']=='test' for r in meta]);variance=run['stats']['pca_variance_ratio']
            value=np.array([np.nan if r[coloring] is None else r[coloring] for r in meta])
            if coloring=='diagnosis':
                for k,label in enumerate(LABELS):
                    mask=test&(value==k);ax.scatter(*xy[mask].T,c=COLORS[k],s=6,alpha=.5,label=label,rasterized=True)
                ax.legend(fontsize=7)
            else:
                mask=test&np.isfinite(value)
                limits={'vmin':0,'vmax':30} if coloring=='mmse' else {'vmin':float(value[test].min()),'vmax':float(value[test].max())}
                sc=ax.scatter(*xy[mask].T,c=value[mask],s=6,alpha=.65,cmap='viridis',rasterized=True,**limits)
                fig.colorbar(sc,ax=ax,label='Observed MMSE' if coloring=='mmse' else 'Age (years)')
            ax.set(title=f"{run['title']} | {run['stats']['dimensions']}D",xlabel=f"Own PC1 ({variance[0]*100:.1f}%)",ylabel=f"Own PC2 ({variance[1]*100:.1f}%)")
        save(fig,out,'comparison_state_'+coloring)
    for mode in ['smoothed','raw','regions']:
        fig,axes=plt.subplots(3,3,figsize=(17,15))
        for ax,run in zip(axes.flat,runs):
            a=run['arrays'];field=run['field'];meta=run['metadata'];test=np.array([r['split']=='test' for r in meta]);xy=a['coordinates']
            ax.scatter(*xy[test].T,c='#bbb',s=2,alpha=.2,rasterized=True)
            support=field['supported']
            if mode=='regions':
                from matplotlib.colors import to_rgb
                color=field['class_probability'][support]@np.array([to_rgb(c) for c in COLORS])
                ax.scatter(*field['grid'][support].T,c=color.clip(0,1),s=18,marker='s',alpha=.8)
            else:
                points=xy[test] if mode=='raw' else field['grid'][support]
                velocity=a['projected_velocities'][test] if mode=='raw' else field['velocity'][support]
                if len(points):
                    speed=np.linalg.norm(velocity,axis=-1);gain=run['stats']['raw_arrow_display_gain']
                    q=ax.quiver(*points.T,*velocity.T,speed,cmap='plasma',norm=Normalize(0,max(float(speed.max()),1e-20)),angles='xy',scale_units='xy',scale=1/gain,width=.002,alpha=.75)
                    fig.colorbar(q,ax=ax,label='Own projected units / year')
                else:ax.text(.5,.5,'No supported grid points',transform=ax.transAxes,ha='center')
            ax.set(title=f"{run['title']}\n{run['stats']['velocity_energy_retained_2d']*100:.1f}% velocity energy retained",xlabel='Own train PC1',ylabel='Own train PC2')
        save(fig,out,'comparison_velocity_'+mode)
    # Give the per-model smoothed fields the same zero-based color convention.
    for run in runs:
        field=run['field'];test=np.array([r['split']=='test' for r in run['metadata']]);xy=run['arrays']['coordinates'];support=field['supported']
        points=field['grid'][support];velocity=field['velocity'][support]
        fig,axes=plt.subplots(1,2,figsize=(13,5.5));axes[0].scatter(*xy[test].T,c='#bbb',s=2,alpha=.2,rasterized=True)
        if len(points):
            speed=np.linalg.norm(velocity,axis=-1)
            q=axes[0].quiver(*points.T,*velocity.T,speed,cmap='plasma',norm=Normalize(0,max(float(speed.max()),1e-20)),angles='xy',scale_units='xy',scale=1/run['stats']['raw_arrow_display_gain'],width=.0025)
            fig.colorbar(q,ax=axes[0],label='Own projected units / year')
        axes[0].set_title(run['title']+' | Training-neighborhood mean velocity')
        region=field['class_probability'][support]@np.array([to_rgb(c) for c in COLORS])
        axes[1].scatter(*points.T,c=region.clip(0,1),marker='s',s=65,alpha=.8)
        axes[1].set_title('Training diagnosis composition: CN / MCI / Dementia')
        for ax in axes:ax.set(xlabel='Own train PC1',ylabel='Own train PC2')
        save(fig,run['folder'],'velocity_field')
    fig,axes=plt.subplots(3,3,figsize=(17,15))
    for ax,run in zip(axes.flat,runs):
        a=run['velocity_plane'];support=a['field_supported'];test=np.array([r['split']=='test' for r in run['metadata']])
        xy=a['coordinates'];points=a['field_grid'][support];velocity=a['field_velocity'][support]
        ax.scatter(*xy[test].T,c='#bbb',s=2,alpha=.2,rasterized=True)
        if len(points):
            speed=np.linalg.norm(velocity,axis=-1);span=max(float(np.ptp(np.quantile(xy,[.01,.99],axis=0),axis=0).max()),1e-10)
            gain=.06*span/max(float(np.quantile(speed,.95)),1e-20)
            q=ax.quiver(*points.T,*velocity.T,speed,cmap='plasma',norm=Normalize(0,max(float(speed.max()),1e-20)),angles='xy',scale_units='xy',scale=1/gain,width=.0025)
            fig.colorbar(q,ax=ax,label='Own projected units / year')
        ax.set(title=f"{run['title']}\n{run['velocity_stats']['velocity_energy_retained_2d']*100:.2f}% velocity energy retained",xlabel='Train velocity axis 1',ylabel='Train velocity axis 2')
    save(fig,out,'comparison_velocity_aligned')
    for case_id in range(3):
        fig,axes=plt.subplots(3,3,figsize=(17,15))
        for ax,run in zip(axes.flat,runs):
            path=run['paths'][case_id];xy=run['arrays']['coordinates'];test=np.array([r['split']=='test' for r in run['metadata']])
            ax.scatter(*xy[test].T,c='#ccc',s=2,alpha=.12,rasterized=True)
            for j,part in enumerate(path['ode_segments']):ax.plot(*np.array(part).T,c='#2171b5',lw=1.5,label='History ODE' if j==0 else None)
            for j,jump in enumerate(path['updates']):ax.plot(*np.array(jump).T,'--',c='#e87b20',lw=1.3,label='Observation update' if j==0 else None)
            ax.scatter(*np.array(path['post_update']).T,c='black',s=12,zorder=4)
            ax.plot(*np.array(path['forecast_5y']).T,c='#8b4ba8',lw=1.6,label='5y pure ODE')
            ax.scatter(*path['origin'],marker='*',s=65,c='#e87b20',zorder=5,label='Initial estimate')
            ax.set(title=run['title'],xlabel='Own train PC1',ylabel='Own train PC2');ax.legend(fontsize=6)
        fig.suptitle(f"Same {runs[0]['paths'][case_id]['label']} example: {runs[0]['paths'][case_id]['subject_id']}")
        save(fig,out,'comparison_case_'+str(case_id+1))
        fig,axes=plt.subplots(3,3,figsize=(17,15))
        for ax,run in zip(axes.flat,runs):
            path=run['velocity_paths'][case_id];xy=run['velocity_plane']['coordinates'];test=np.array([r['split']=='test' for r in run['metadata']])
            ax.scatter(*xy[test].T,c='#ccc',s=2,alpha=.12,rasterized=True)
            for j,part in enumerate(path['ode_segments']):ax.plot(*np.array(part).T,c='#2171b5',lw=1.5,label='History ODE' if j==0 else None)
            for j,jump in enumerate(path['updates']):ax.plot(*np.array(jump).T,'--',c='#e87b20',lw=1.3,label='Observation update' if j==0 else None)
            ax.scatter(*np.array(path['post_update']).T,c='black',s=12,zorder=4)
            ax.plot(*np.array(path['forecast_5y']).T,c='#8b4ba8',lw=1.6,label='5y pure ODE')
            ax.scatter(*path['origin'],marker='*',s=65,c='#e87b20',zorder=5)
            ax.set(title=run['title'],xlabel='Train velocity axis 1',ylabel='Train velocity axis 2');ax.legend(fontsize=6)
        save(fig,out,'comparison_velocity_case_'+str(case_id+1))
    line_colors=['#8c564b','#1f77b4','#d62728','#ff7f0e','#9467bd','#7f7f7f','#17becf','#2ca02c','#111111']
    fig,axes=plt.subplots(1,3,figsize=(18,5.8))
    for case_id,ax in enumerate(axes):
        case=runs[0]['forecasts'][case_id]
        for color,run in zip(line_colors,runs):
            row=run['forecasts'][case_id];ax.plot(row['years'],row['mmse_curve'],c=color,lw=1.5,label=run['title'])
            if run['family']=='healthflux':
                ax.scatter(0,row['observed_cutoff_prediction'],c=color,s=35,zorder=5)
                ax.scatter(0,row['event_free_cutoff_prediction'],facecolors='white',edgecolors=color,s=30,zorder=5)
            if run['family']=='ode_rnn':
                xx=np.array(row['visit_years']);yy=np.array(row['native_visit_mmse']);mask=(xx>0)&(xx<=5)
                ax.plot(xx[mask],yy[mask],'s--',c=color,lw=1.5,ms=4,label='ODE-RNN native feedback')
        xx=np.array(case['visit_years']);actual=np.array([np.nan if v is None else v for v in case['actual_visit_mmse']]);mask=(xx>=0)&(xx<=5)&np.isfinite(actual)
        ax.scatter(xx[mask],actual[mask],marker='*',c='black',s=70,zorder=6,label='Observed MMSE')
        ax.set(title=f"{case['label']} example | {case['subject_id']}",xlabel='Years after 3rd observed visit',ylabel='MMSE (unclipped)');ax.legend(fontsize=6.8)
    save(fig,out,'comparison_case_mmse')
    hfx=next(r for r in runs if r['family']=='healthflux');r384=np.load(hfx['folder']/'health_representation_384.npz')
    fig,axes=plt.subplots(2,3,figsize=(17,10));meta=hfx['metadata'];test=np.array([r['split']=='test' for r in meta])
    for i,(xy,title) in enumerate([(hfx['arrays']['coordinates'],'ODE state h: 256D'),(r384['coordinates'],'Observed decoder representation r: 384D')]):
        for j,key in enumerate(['diagnosis','mmse','age']):
            value=np.array([np.nan if row[key] is None else row[key] for row in meta]);ax=axes[i,j]
            if key=='diagnosis':
                for k,label in enumerate(LABELS):
                    mask=test&(value==k);ax.scatter(*xy[mask].T,c=COLORS[k],s=6,alpha=.5,label=label,rasterized=True)
                ax.legend(fontsize=7)
            else:
                mask=test&np.isfinite(value);sc=ax.scatter(*xy[mask].T,c=value[mask],cmap='viridis',s=6,alpha=.65,rasterized=True);fig.colorbar(sc,ax=ax)
            ax.set(title=title+' | '+key,xlabel='Own train PC1',ylabel='Own train PC2')
    save(fig,out,'healthflux_h_vs_full_representation')


def report(out,runs):
    def f(value,n=3):return '—' if value is None else f'{value:.{n}f}'
    def p(value):return f(value*100,2)+'%'
    lines=['# 八组模型的状态空间、速度场与轨迹比较','',
           '## 1. 本次如何建立共同对照','',
           '我们已经用 shared_matched 六维模型观察了患者状态、ODE 速度和观测更新。本次将同一套流程扩展到 HealthFlux-kl、四组 latent2x、概率 ODE-AE、Latent ODE 和 ODE-RNN，并把原 shared_matched 作为第九个参照面板。所有检查点均来自已经完成的 seed 0 实验。','',
           'HealthFlux-kl使用[共同ADNI任务适配](../../../healthflux/README.zh-CN.md)中KL=0.01的paper_kl检查点；Latent ODE和ODE-RNN使用此前native容量组；ODE-AE使用原概率版本。latent2x baseline不加新增损失，both加入0.01×weak_prior与0.01×Jacobian，prior与jacobian分别只加对应一项。参数、检查点轮次和同任务误差见下一节。','',
           '各模型读取完全相同的患者和历史截止点：2,900位患者、17,165个状态，其中训练12,544个、验证2,219个、测试2,402个。每个点代表一次截止历史后的状态估计。同一患者可以贡献多个点。测试状态中有CN 826个、MCI 850个、Dementia 369个，另外357个当次诊断缺失；没有用历史诊断补填。','',
           '每个模型只在自己的训练状态上拟合不做逐维标准化的PCA，再投影验证、测试、速度和轨迹。九宫格中的每幅图使用该模型自己的坐标和标尺；模型之间比较诊断结构、投影保留率、方向一致性和同一患者的轨迹。原六维 shared_matched 直接沿用此前的PCA均值、基向量和方向。','',
           '### 先明确图中的“状态”','',
           '| 模型 | 本次主图的状态 | 原维度 | 观测怎样进入状态 | 速度换算 |',
           '|---|---|---:|---|---|',
           '| HealthFlux-kl | ODE直接推进的确定性状态h | 256 | 先诊断事件、再其他模态，Gaussian后验与GRU更新 | 原十年速度÷10 |',
           '| latent2x四组 | 历史截止处UPDATE后的共享状态z | 28 | 正向逐次吸收历史 | 原速度已是每年 |',
           '| 概率ODE-AE | 从初始后验均值生成、推进到截止时刻的状态z | 6 | 反向编码历史后，生成状态轨迹 | 原速度已是每年 |',
           '| Latent ODE | 历史截止处生成后验的均值z | 32 | 64维编码器吸收历史，产生32维生成状态 | 原十年速度÷10 |',
           '| ODE-RNN | 当前观测更新后的GRU隐藏状态h | 32 | 正向ODE与观测GRU交替 | 原十年速度÷10 |',
           '| shared_matched参照 | 历史截止处UPDATE后的共享状态z | 6 | 正向逐次吸收历史 | 原速度已是每年 |','',
           'HealthFlux另保存完整384维解码表示r=LN([h,z])的状态图，见第5节。主速度图使用256维h，因为它是直接被ODE推进的位置。其速度在每条状态自己的实际年龄和背景条件下计算。','',
           '## 2. 先看同一任务上的定量结果','',
           '全部模型都重新在GPU上运行原验证和测试评分，并与已保存指标核对。下面的历史、未来与整体误差均是原40字段MSE与类别交叉熵按字段汇总的结果。它们沿用原训练选点，不根据本次可视化重新选择检查点。','',
           '| 模型 | 可训练参数 | 总参数 | 最佳轮次 | 验证整体 | 测试历史重构/预测 | 测试未来 | 测试整体 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        s=run['stats'];m=s['native_metrics'];lines.append(f"| {run['title']} | {s['trainable_parameters']:,} | {s['parameters']:,} | {s['checkpoint_epoch']} | {f(m['validation']['overall']['reconstruction_error'],5)} | {f(m['test']['interpolation']['reconstruction_error'],5)} | {f(m['test']['forecast']['reconstruction_error'],5)} | {f(m['test']['overall']['reconstruction_error'],5)} |")
    lines+=['','在这批固定检查点中，latent2x both的测试未来误差1.00030、整体误差0.70254均最低。用于选择检查点的验证整体误差则是latent2x baseline最低，为0.74185。','',
            'ODE-RNN的历史项是读取当前观测前的因果预测；其余模型的历史项利用截止历史进行重构。未来列使用各模型原生评测路径，其中ODE-RNN包含预测反馈更新。后面的纯ODE轨迹则把连续动力学单独取出观察。','',
            '### 二维图保留了多少信息','',
            '状态保留率衡量两维承载的训练状态方差。速度保留率在全部17,165个状态上统计，衡量真实高维速度在这两维中的平方和占比，与原报告口径一致。有效秩根据训练状态方差谱计算，用来描述状态实际占用了多少变化方向。最后一列对每位测试患者只取最后状态，在有真实诊断的354人上计算高维轮廓系数：正值越高，表示同类距离相对更小、异类距离相对更大。','',
            '| 模型 | 状态维度 | 两维状态方差 | 两维速度能量 | 状态有效秩 | 局部投影方向一致性中位数 | 最后状态诊断轮廓系数 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        s=run['stats'];lines.append(f"| {run['title']} | {s['dimensions']} | {p(s['state_variance_retained_2d'])} | {p(s['velocity_energy_retained_2d'])} | {f(s['effective_state_rank'],2)} | {f(s['local_direction_coherence_median'],4)} | {f(s['last_patient_diagnosis_silhouette'],4)} |")
    lines+=['','从完整状态的变化方向看，latent2x四组的有效秩为11.06～13.40，高于六维参照的5.23。扩容后的状态确实使用了更多变化方向，而前两维覆盖其中约39%～47%。','',
            '## 3. 状态空间怎样保留诊断与临床信息','',
            '所有模型都使用相同颜色：CN为绿色，MCI为黄色，Dementia为红色。诊断和MMSE都是已有观测字段；这些图检查模型是否在状态中保留了相应临床信息。','',
            '![按真实诊断着色的九模型状态空间](comparison_state_diagnosis.png)','',
            '![按真实MMSE着色的九模型状态空间](comparison_state_mmse.png)','',
            '![按年龄着色的九模型状态空间](comparison_state_age.png)','',
            'latent2x四组都保留了明显的CN、MCI、Dementia分组和交界。jacobian组的高维诊断轮廓系数最高，为0.2359；both组为0.1821，却有更低的测试未来误差。疾病分组清楚和未来字段预测准确，分别描述了表示与动力学的不同表现。','',
            '概率ODE-AE的前两维已经覆盖99.95%的状态变化，三种诊断仍明显交叠，高维诊断轮廓系数为−0.0392。Latent ODE前两维保留82.12%，图中也存在较多诊断混合。HealthFlux的主图则呈分支条带状，诊断分组在这些条带上交错分布。','',
            '## 4. 同时比较平滑场与每个真实状态的原始速度','',
            '先在每个完整状态上计算真实速度，再乘以该模型的PCA矩阵，得到二维速度。HealthFlux、Latent ODE、ODE-RNN的原时间单位是十年，已统一换算为每年。每幅图沿用自身的坐标单位，箭头使用该模型统一的显示缩放，色条显示原坐标单位/年的投影速度。','',
            '### 邻域平均：看整体方向','',
            '在训练状态的PCA平面中建立32×28网格，每个网格取最近30个训练状态的真实速度做加权平均。附近训练样本不足的位置留白。背景是测试状态，场的估计只使用训练状态。方向一致性是平均速度长度除以速度长度的加权平均，取值接近1表示邻域投影方向更一致。','',
            '![九模型局部平均速度场](comparison_velocity_smoothed.png)','',
            '对应网格的诊断组成如下，三类邻居的比例直接混色：','',
            '![速度网格附近的训练诊断组成](comparison_velocity_regions.png)','',
            '### 未平滑：看每个测试状态的实际输出','',
            '下面每支箭头都来自该测试状态自身的高维速度，不做平均。每组模型另外保存训练、验证、测试三幅全量原始箭头图，链接列于第9节。','',
            '![九模型测试状态的原始速度](comparison_velocity_raw.png)','',
            '### 再换一个平面，把主要运动方向显示出来','',
            '状态PCA选择的是患者差异最大的方向；速度主平面选择的是ODE运动最集中的方向。这里对训练状态上的原始速度做未中心化SVD，保留平均流，取前两个方向。验证、测试和病例轨迹沿用同一投影。全部SVD与能量计算使用CUDA float64。','',
            '下表最后一列计算训练平均速度所占的能量比例：N‖平均速度‖²除以各状态速度平方和。接近100%表示采样位置上的速度方向和大小都接近共同值。','',
            '| 模型 | 状态PCA中的速度能量 | 速度主平面中的速度能量 | 该平面保留的状态方差 | 平均速度向量解释的训练速度能量 |',
            '|---|---:|---:|---:|---:|']
    for run in runs:
        v=run['velocity_stats'];lines.append(f"| {run['title']} | {p(run['stats']['velocity_energy_retained_2d'])} | {p(v['velocity_energy_retained_2d'])} | {p(v['state_variance_retained_2d'])} | {p(v['train_mean_velocity_energy_fraction'])} |")
    lines+=['','![速度主平面中的九模型场](comparison_velocity_aligned.png)','',
            'HealthFlux的状态PCA只保留0.82%的速度能量，换到速度主平面后保留约100%；latent2x四组从7.60%～14.16%提高到93.18%～98.15%。这说明它们的主要运动方向与最大状态差异方向并不重合。','',
            'HealthFlux-kl、Latent ODE、ODE-RNN的平均速度向量解释了几乎100%的训练速度能量，概率ODE-AE也超过99.99%。在本次采样的训练状态上，这些ODE速度接近共同方向的平移。latent2x四组的对应比例约为72.58%～83.94%，还保留了更多随位置改变的速度分量。','',
            '这也解释了为什么局部方向一致性接近1不直接对应最低预测误差：一组几乎恒定的速度也会高度一致。测试预测还取决于状态中保留的信息、观测更新、读出函数及原生未来反馈路径。','',
            '## 5. HealthFlux的256维ODE位置与384维完整健康表示','',
            'HealthFlux的h由ODE连续推进；随机部分z在有观测时取后验均值，在无观测查询时取先验均值。解码器使用二者经LayerNorm组成的384维表示。因此，本次提供h和完整表示r两套训练集PCA：前者配速度箭头，后者观察完整健康表示的临床信息。','',
            '![HealthFlux的确定性位置与完整表示](healthflux_h_vs_full_representation.png)','',
            '## 6. 固定同三位患者，看ODE与观测更新如何交替','',
            '病例沿用原shared_matched报告中的三位测试患者：CN示例003_S_4900、MCI示例002_S_4219、Dementia示例002_S_0619。类别来自完整记录的末次真实诊断，仅用于固定绘图名单。三位患者在所有模型中完全相同。','',
            '蓝线为历史时段的ODE轨迹，橙色虚线为真实观测导致的状态更新，黑点标记检查时刻的状态，紫线为末次状态出发的五年纯ODE外推。星号是这条轨迹的初始估计。','',
            '对正向模型，一条历史轨迹固定一次初始化，再依次吸收观测。shared_matched系列按原实现使用该次完整历史的截止年龄初始化。对概率ODE-AE和Latent ODE，主图使用生成状态空间，蓝线是整段历史条件下的生成轨迹；它们的观测编码器位于另一状态空间，因此主图没有橙色更新线。','',
            '![同一CN示例的九模型轨迹](comparison_case_1.png)','',
            '![同一MCI示例的九模型轨迹](comparison_case_2.png)','',
            '![同一Dementia示例的九模型轨迹](comparison_case_3.png)','',
            '同一批轨迹也在速度主平面中投影：[CN病例](comparison_velocity_case_1.png) · [MCI病例](comparison_velocity_case_2.png) · [Dementia病例](comparison_velocity_case_3.png)。这组图更直接显示主要ODE位移。','',
            '## 7. 冻结前三次检查，比较后续MMSE曲线','',
            '这一步与上一节的完整历史展示分开：每个模型只读取同一患者的前三次检查，然后在之后五年内只沿ODE推进，曲线读取各自已有的MMSE输出。真实随访仅用于黑色星号。没有重新拟合临床预测头。','',
            'HealthFlux在0时刻用实心点标记观测后读出，用空心点标记同一h位置下的无观测先验读出；未来曲线从先验分支出发。ODE-RNN另外用绿色虚线方块画出原生协议在真实未来检查时点的预测反馈结果，实线展示纯ODE分量。','',
            '![相同前三次历史下的未来MMSE比较](comparison_case_mmse.png)','',
            '密集绘图查询不会替换真实检查间隔上的积分端点。HealthFlux仍使用原RK4，其他模型仍使用原自适应求解器；未来检查时点的读出已与原forward数值核对。','',
            'CN示例中，shared_matched和latent2x的输出更接近29～30分，HealthFlux起点输出约32.28分，概率ODE-AE和Latent ODE则偏低。这里保留各头的原始输出，没有裁剪到量表范围。MCI示例中，多数组模型给出缓慢变化；HealthFlux的起点偏低，并继续下降。Dementia示例的真实后续MMSE由19升到22，而几组shared模型仍预测下降，这个病例展示了群体下降趋势与个体随访改善之间的差异。','',
            'ODE-RNN的纯ODE MMSE实线在三例中几乎水平，原生反馈预测另用虚线方块显示。它的最终预测行为需要把ODE与反馈GRU一起看。HealthFlux的读出还经过年龄、背景条件的随机先验和非线性解码器，因此相近的潜在速度也可以在不同起点读出不同MMSE曲线。','',
            '## 8. 将几类发现放在一起比较','',
            '先看表示：latent2x的四组在完整状态中使用了更多变化方向，并保持了疾病相关结构。再看动力学：其速度变化主要集中在少数方向，但仍随状态位置变化；both组的前两速度方向保留98.15%的能量，baseline为93.18%。最后回到共同任务，both组的未来与整体测试误差最低。','',
            'jacobian组有本次最高的诊断轮廓系数，但未来误差为1.02670，略高于baseline的1.02030。状态分组指标与预测指标没有形成同一排序。概率ODE-AE、Latent ODE和HealthFlux-kl的场可以非常齐整，测试误差仍高于latent2x，这说明图中的平滑和集中需要与实际预测结果一起解读。','',
            '这批seed 0结果形成了清楚的对应关系：latent2x增加了状态变化的有效方向，both组让速度更集中在主要方向并得到较低测试误差；诊断分群和局部方向一致性，则分别描述了表示结构和场的局部行为。','',
            '## 9. 逐模型查看与复现记录','',
            '| 模型 | 状态 | 平滑场 | 原始全量场 | 速度主平面 | 患者轨迹 | 交互页面 | GPU记录 |',
            '|---|---|---|---|---|---|---|---|']
    for run in runs:
        name=run['name'];lines.append(f"| {run['title']} | [图]({name}/state_space.png) | [图]({name}/velocity_field.png) | [图]({name}/velocity_field_raw.png) | [图]({name}/velocity_aligned.png) | [图]({name}/patient_paths.png) | [打开]({name}/interactive.html) | [记录]({name}/statistics.json) |")
    lines+=['','全部模型保持原检查点参数，未重新训练。提取、ODE求解、PCA、近邻和平面投影均在RTX 4090上执行；CPU用于汇总与绘图。每组记录物理GPU、实际张量设备、检查点与数据哈希、原指标重算差异及状态提取核对。','',
            '[任务与检查点清单](manifest.json) · [完成核对](verification.json) · [汇总数据](summary.json)','',
            '```bash','bash scripts/run_state_space_model_comparison.sh','```','']
    summary={'runs':[{k:run[k] for k in ['name','title','family']}|{'statistics':run['stats']} for run in runs]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'report.zh-CN.md').write_text('\n'.join(lines))


def generate(out):
    out=Path(out);runs=read_runs(out)
    if len(runs)!=len(RUNS):raise RuntimeError(f'Need all {len(RUNS)} completed models; got {len(runs)}')
    figures(out,runs);report(out,runs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default=str(OUT));generate(parser.parse_args().output)
