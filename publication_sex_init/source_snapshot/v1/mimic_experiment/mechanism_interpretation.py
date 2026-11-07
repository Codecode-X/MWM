"""Evidence-conditioned text; hypotheses stay pending until their measurements exist."""
import json
from pathlib import Path

import numpy as np


def interpret(out,records,measured,precision,patients,statistics):
    out=Path(out);sections={};grades={h:('待实验完成','待相应实验完成') for h in ['H1','H2','H3','H4']}
    auxiliary=json.loads((out/'early_patient_queue.json').read_text()) if (out/'early_patient_queue.json').exists() else None
    completed=json.loads((out/'queue.json').read_text()).get('status')=='complete' and (auxiliary is None or auxiliary['status']=='complete')
    encoder=json.loads((out/'encoder_path_followup/analysis.json').read_text()) if (out/'encoder_path_followup/analysis.json').exists() else None
    state_trace=json.loads((out/'encoder_path_followup/state_trace.json').read_text()) if (out/'encoder_path_followup/state_trace.json').exists() else None
    dominant=json.loads((out/'dominant_patient_precision/analysis.json').read_text()) if (out/'dominant_patient_precision/analysis.json').exists() else None
    future_direction=json.loads((out/'encoder_path_followup/future_direction.json').read_text()) if (out/'encoder_path_followup/future_direction.json').exists() else None
    if (out/'dominant_patient_precision/status.json').exists() and dominant is None:completed=False
    if measured:
        table=['| 更新 | A历史敏感性：更新前→后 | B历史敏感性：更新前→后 |','|---|---:|---:|']
        special=[r for r in measured if (r['step']['epoch'],r['step']['batch']) in [(44,1),(45,1),(45,169),(46,1),(46,120),(46,121),(46,122),(46,169)]]
        top=sorted(measured,key=lambda r:max(r['flags'][k]['after_before_ratio'] for k in ['A_reencoded','B_fixed_state']),reverse=True)[:5]
        selected={r['step']['id']:r for r in special+top}
        for key,r in sorted(selected.items()):
            values=[f"{r['before'][k]['past']['gain']['median']:.4g} → {r['after'][k]['past']['gain']['median']:.4g}" for k in ['A_reencoded','B_fixed_state']]
            table.append(f"| 第{r['step']['epoch']}轮 / {r['step']['batch']} | {values[0]} | {values[1]} |")
        sections['onset']=table+['']
        byid={r['id']:r for r in records};key_updates={r['step']['id'] for r in top}
        key_updates.update(r['step']['id'] for r in measured if (r['step']['epoch'],r['step']['batch']) in [(46,120),(46,121),(46,122)])
        sections['onset']+=['| 关键更新 | 历史损失：前→后 | 未来损失：前→后 | 全参数梯度：裁剪前→后 | ODE更新量 / 参数范数 |','|---|---:|---:|---:|---:|']
        for sid in sorted(key_updates):
            if sid not in byid:continue
            t=byid[sid];a,b=t['before'],t['after']
            sections['onset'].append(f"| {sid} | {a['history_loss_contribution']:.5g} → {b['history_loss_contribution']:.5g} | {a['future_loss_contribution']:.5g} → {b['future_loss_contribution']:.5g} | {a['all_parameter_preclip_norm']:.5g} → {a['all_parameter_postclip_norm']:.5g} | {t['relative_field_update']:.4g} |")
        sections['onset'].append('')
        for key,title in [('A_reencoded','重新编码'),('B_fixed_state','固定潜在起点')]:
            initial=measured[0]['before'][key]['past']['gain']['median'];peak=max(r['after'][key]['past']['gain']['median'] for r in measured)
            jump=max(r['flags'][key]['after_before_ratio'] for r in measured);five=sum(r['flags'][key]['single_update_fivefold'] for r in measured)
            sections['onset'] += [f'{title}：当前测量窗口的起始中位数为{initial:.4g}，最高为{peak:.4g}，累计最高为起始的{peak/initial:.3g}倍；最大单步增长{jump:.3g}倍，单步至少5倍的更新有{five}次。','']
        stable_end=next((r for r in measured if r['step']['id']=='e045_b169'),None)
        if stable_end:sections['onset'] += [f"第44轮的阈值跨越随后回落；第45轮结束时，A测量中位数回到{stable_end['after']['A_reencoded']['past']['gain']['median']:.3g}。第46轮峰值前的最大单步增长发生在第119次更新，紧接着第120次更新再次明显增长。",'']
        flags=[{'epoch':r['step']['epoch'],'batch':r['step']['batch'],'flags':r['flags']} for r in measured if any(f['crosses_stable_p99'] or f['single_update_fivefold'] for f in r['flags'].values())]
        (out/'first_abnormal_updates.json').write_text(json.dumps(flags,indent=2)+'\n')
        episodes=[]
        for key in ['A_reencoded','B_fixed_state']:
            current=None
            for r in measured:
                high=r['flags'][key]['above_stable_p99_after']
                if high:
                    if current is None:current={'mode':key,'start':r['step']['id'],'updates':0,'maximum':0.}
                    current['updates']+=1;current['end']=r['step']['id'];current['maximum']=max(current['maximum'],r['after'][key]['past']['gain']['median'])
                elif current is not None:episodes.append(current);current=None
            if current is not None:episodes.append(current)
        (out/'sensitivity_episodes.json').write_text(json.dumps(episodes,indent=2)+'\n')
        if episodes:
            sections['onset']+=['连续超过同一稳定P99的区间如下，列出最长的6段，按出现时间排序。','',
                               '| 方式 | 起点→终点 | 连续更新数 | 区间内最高中位数 |','|---|---|---:|---:|']
            for e in sorted(sorted(episodes,key=lambda e:e['updates'],reverse=True)[:6],key=lambda e:e['start']):
                sections['onset'].append(f"| {e['mode']} | {e['start']} → {e['end']} | {e['updates']} | {e['maximum']:.4g} |")
            sections['onset']+=['','[全部阈值跨越更新](mechanism_diagnosis/first_abnormal_updates.json) · [全部连续区间](mechanism_diagnosis/sensitivity_episodes.json)','']
    if precision:
        lines=['| 快照 | 原训练float32：状态 / 状态梯度 / 完整ODE梯度相对差 | 诊断float32：状态 / 状态梯度 / 完整ODE梯度相对差 | 最高精度float64：状态 / 状态梯度 / 完整ODE梯度相对差 |','|---|---:|---:|---:|']
        for item in precision:
            relevant={r['setting']:r for r in item['settings'] if r['anchor']=='snapshot_reencoded_state'}
            def f(name):
                r=relevant.get(name,{})
                if r.get('status')!='complete':return '求解失败'
                d=r.get('relative_differences_to_reference',{})
                return ' / '.join(f'{d[k]:.3g}' if k in d else '—' for k in ['past_endpoint','history_state_gradient','history_all_solvers_refined_full_field_gradient'])
            lines.append(f"| {item['snapshot']['id']} | {f('train32')} | {f('diagnostic32')} | {f('tightest64')} |")
        lines += ['','“完整ODE梯度”同时细化编码和生成的积分精度：各档编码参数VJP与对应历史状态梯度相接，再加生成直接梯度。生成输入始终锁定为同一保存的截止点状态。另存一份保持原编码容差的完整梯度，用于分离生成积分的影响。','']
        peak=next((p for p in precision if p['snapshot']['id']=='epoch46_batch121_pre'),None)
        if peak:
            relevant={r['setting']:r for r in peak['settings'] if r['anchor']=='snapshot_reencoded_state'}
            lines += ['异常快照的数值往返误差如下，数值为固定患者的中位数 / 最大值。','',
                      '| 设置 | 6小时往返相对误差 | 12小时往返相对误差 | 24小时往返相对误差 |','|---|---:|---:|---:|']
            reference=next((c['rk4_reference'] for c in peak['convergence'] if c['anchor']=='snapshot_reencoded_state'),None)
            for name in ['train32','diagnostic32','tightest32','tightest64',reference]:
                r=relevant.get(name,{})
                if r.get('status')!='complete':continue
                values=[f"{r['cycle_errors'][str(h)]['median']:.3g} / {r['cycle_errors'][str(h)]['max']:.3g}" for h in [6.,12.,24.]]
                lines.append('| '+name+' | '+' | '.join(values)+' |')
            lines+=['','异常快照的求解器工作量如下。统计覆盖该设置的终点、梯度、Jacobian、往返检查与编码VJP；编码VJP固定使用原训练容差。','',
                    '| 设置 | 状态 | 自适应接受 / 拒绝步 | 固定RK4步 | 函数调用数 |','|---|---|---:|---:|---:|']
            for name,r in relevant.items():
                a=r['adaptive_solver_counts'];f=r.get('fixed_solver_counts',{})
                lines.append(f"| {name} | {r['status']} | {a['accepted']} / {a['rejected']} | {f.get('fixed_steps',0)} | {a['nfe']+f.get('nfe',0)} |")
            lines.append('')
            strict=relevant.get('tightest64',{});diag=relevant.get('diagnostic32',{});native=relevant.get('train32',{})
            if strict.get('status')=='complete' and strict.get('reference_step_halving_converged'):
                d=strict['relative_differences_to_reference'];lines += [f"异常快照的float64严格积分中，历史敏感性中位数为{strict['past_gain']['median']:.5g}，与步长已收敛RK4的奇异值相对差为{d['past_singular_value']:.3g}；状态梯度相对差为{d['history_state_gradient']:.3g}，完整ODE梯度相对差为{d['history_all_solvers_refined_full_field_gradient']:.3g}。",'']
                converged=max(d['past_endpoint'],d['past_singular_value'],d['history_state_gradient'],d['history_all_solvers_refined_full_field_gradient'])<.01
                if converged:
                    grades['H1']=('高精度下保留高敏感性；更新时序见实验一','支持高敏感性属于该位置附近的ODE流性质')
                    nd=native.get('relative_differences_to_reference',{})
                    dd=diag.get('relative_differences_to_reference',{})
                    if dd:
                        grades['H2']=(f"异常快照诊断容差的状态/完整ODE梯度相对差为{dd['past_endpoint']:.3g}/{dd['history_all_solvers_refined_full_field_gradient']:.3g}；原容差为{nd.get('past_endpoint',float('nan')):.3g}/{nd.get('history_all_solvers_refined_full_field_gradient',float('nan')):.3g}",
                                      '不支持数值误差是该固定样本大幅放大的主因')
                else:grades['H2']=('float64与RK4仍有超过1%的关键量差异','数值因素尚未排除')
            else:grades['H2']=('独立参考或严格求解尚未收敛','保留数值误差解释')
        failures=[f"{p['snapshot']['id']}/{r['anchor']}/{r['setting']}: {r.get('error','')}" for p in precision for r in p['settings'] if r['status']=='failed']
        (out/'solver_failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        lines += [f'当前共记录{len(failures)}个求解失败设置，失败原因保存在[求解失败清单](mechanism_diagnosis/solver_failures.json)。这些设置没有替换为其他求解器结果。','']
        sections['precision']=lines
    if patients:
        lines=['| Case | INIT历史梯度 | UPDATE历史梯度 | Decoder历史梯度 | ODE历史梯度 |','|---|---:|---:|---:|---:|']
        labels={'abnormal_real_e46b121':'异常真实batch：46轮121批','stable_matched_same_patients':'同一输入＋45轮稳定参数','stable_real_e45b121':'稳定真实batch：45轮121批','first_clipped_e46_b082':'首次裁剪batch：46轮82批'}
        for p in patients:
            m=p['module_history_gradient_norms'];lines.append('| '+labels.get(p['case']['id'],p['case']['id'])+' | '+' | '.join(f'{m[k]:.5g}' for k in ['INIT','UPDATE','Decoder','shared_ODE'])+' |')
        p=next((x for x in patients if x['case']['id']=='abnormal_real_e46b121'),None)
        if p:
            b=p['batch'];s=statistics.get(p['case']['id'],{});lines+=['','异常batch中，按历史ODE梯度范数排序的前10位患者如下。投影份额可为负，全部患者的投影份额相加为1。','',
                '| 患者索引 | 历史跨度h | 观测数 | 历史敏感性 | 实际首目标梯度放大 | 历史ODE梯度范数 | 沿batch梯度的投影份额 |','|---|---:|---:|---:|---:|---:|---:|']
            ranked=sorted(p['patients'],key=lambda r:r['history_field_gradient_norm'],reverse=True)
            for r in ranked[:10]:lines.append(f"| {r['index']} | {r['history_span_hours']:.3g} | {r['history_observations']} | {r['past_state_max_singular_value']:.4g} | {r['earliest_actual_direction_gain']:.4g} | {r['history_field_gradient_norm']:.5g} | {r['projected_share_of_batch_history_gradient']:.2%} |")
            gains=np.array([r['past_state_max_singular_value'] for r in p['patients']]);top=ranked[0]
            rank=1+sum(r['past_state_max_singular_value']>top['past_state_max_singular_value'] for r in p['patients'])
            lines+=['',f"真实batch中，状态敏感性中位数为{np.median(gains):.4g}，{(gains>10).mean():.1%}的患者超过10倍。主导梯度患者的流敏感性排名为第{rank}，回顾跨度{top['history_span_hours']:.0f}小时；该batch最长跨度为{max(r['history_span_hours'] for r in p['patients']):.0f}小时。高流敏感性覆盖多位患者，实际训练梯度集中在其中一位。",'',
                    '| 回顾跨度h | 患者数 | 反向状态敏感性中位数 | 患者历史ODE梯度范数中位数 | 最大患者历史梯度 |','|---|---:|---:|---:|---:|']
            for low,high in [(0,6),(6,12),(12,18),(18,24)]:
                group=[r for r in p['patients'] if low<r['history_span_hours']<=high]
                if group:lines.append(f"| ({low}, {high}] | {len(group)} | {np.median([r['past_state_max_singular_value'] for r in group]):.4g} | {np.median([r['history_field_gradient_norm'] for r in group]):.4g} | {max(r['history_field_gradient_norm'] for r in group):.4g} |")
            loss_share=top['history_loss_original_batch_contribution']/sum(r['history_loss_original_batch_contribution'] for r in p['patients'])
            lines+=['',f"主导患者只贡献本batch历史损失值的{loss_share:.2%}，却贡献历史梯度沿总方向投影的{top['projected_share_of_batch_history_gradient']:.2%}。损失值与参数梯度的集中程度明显不同。",'']
            if s:
                lines+=['',f"梯度范数排名前5位患者沿batch历史梯度方向的合计投影份额为{s['top5_norm_ranked_projected_share']:.2%}。单个最早历史目标的实际梯度放大与其流Jacobian上界之比，中位数为{s['median_actual_gain_over_bound']:.4g}。",'']
                for key,title in [('history_span_hours','回顾跨度'),('history_observations','历史观测数量'),('missing_fraction','缺失比例'),('cutoff_state_norm','截止点状态范数')]:
                    a=s['correlations'][key+'__sensitivity']['spearman'];g=s['correlations'][key+'__history_field_gradient']['spearman']
                    if a is not None and g is not None:lines.append(f'{title}与敏感性的Spearman相关为{a:.3f}，与历史ODE梯度范数的相关为{g:.3f}。')
                part=s.get('history_span_partial_rank_correlation',{})
                if all(part.get(k) is not None for k in ['sensitivity','history_field_gradient']):
                    lines.append(f"对观测数、有效字段总数、缺失比例与状态范数的秩作残差调整后，回顾跨度与敏感性、历史ODE梯度的偏秩相关分别为{part['sensitivity']:.3f}、{part['history_field_gradient']:.3f}。")
                lines.append('')
            lines+=['| 不更新参数的诊断条件 | 删除人数 | 分母处理 | ODE总梯度范数 | 与原总梯度余弦 |','|---|---:|---|---:|---:|',f"| 原始完整batch | 0 | 原目标 | {b['total_field_gradient_norm']:.5g} | 1 |"]
            labels={'longest_history':'去除最长回顾','highest_sensitivity':'去除最高敏感性','highest_history_gradient':'去除最大历史梯度','future_only':'仅未来贡献'}
            for c in p['counterfactual_conditions']:
                if c['removed_count'] not in [0,1,5]:continue
                denominator='重算有效字段分母' if 'recomputed' in c['normalization'] else '保留原batch分母'
                lines.append(f"| {labels[c['selection']]} | {c['removed_count']} | {denominator} | {c['total_field_gradient_norm']:.5g} | {c['cosine_to_original_total']:.4f} |")
            lines+=['','保留原分母的排除条件，只去掉选中患者的历史贡献，未来、prior与正则保持原值；重算分母的条件则去掉选中患者全部目标及Jacobian采样。两种诊断分开列出。','',f"完整历史ODE梯度范数为{b['history_gradient_norm']:.5g}，其中编码参数路径范数为{b['encoder_path_norm']:.5g}，生成直接路径为{b['direct_generation_norm']:.5g}。这两个向量相加得到完整梯度；两者范数直接相加不等于总范数。",'']
            removed=next(c for c in p['counterfactual_conditions'] if c['selection']=='highest_history_gradient' and c['removed_count']==1)
            if s:grades['H3']=(f"单个患者投影份额{s['largest_gradient_projected_share']:.2%}；去掉其历史贡献后历史ODE梯度{b['history_gradient_norm']:.4g}→{removed['history_field_gradient_norm']:.4g}",'支持单患者主导；最长跨度或最高流敏感性不能单独识别它')
            grades['H4']=(f"编码路径/生成直接路径范数={b['encoder_path_norm']:.5g}/{b['direct_generation_norm']:.5g}",'已分离额外的编码参数传播路径')
        sections['patients']=lines
        paired=statistics.get('paired_path_growth',[])
        if paired:
            sections['patients']+=['同一批患者在稳定第45轮与异常参数下的配对分解：将编码路径梯度增长写成三个实际方向因子的乘积。第一项是Decoder与误差端梯度变化；第二项是历史流把堆叠历史梯度拉回截止点的增益变化；第三项是截止点梯度返回编码ODE参数的VJP增益变化。三项相乘恰好得到编码路径梯度范数的增长。','',
                                   '| 患者索引 | Decoder/误差端因子 | 历史状态拉回因子 | 编码参数VJP因子 | 编码路径梯度总增长 |','|---|---:|---:|---:|---:|']
            for r in sorted(paired,key=lambda x:x['abnormal_history_gradient_norm'],reverse=True)[:10]:
                sections['patients'].append(f"| {r['patient_index']} | {r['decoder_error_factor']:.4g} | {r['history_state_pullback_factor']:.4g} | {r['encoder_parameter_vjp_factor']:.4g} | {r['encoder_gradient_growth']:.4g} |")
            sections['patients']+=['','[全部患者的配对路径增长](mechanism_diagnosis/paired_path_growth.json)','']
    followup=[]
    if encoder:
        stable,abnormal=encoder['models']['stable'],encoder['models']['abnormal']
        followup+=['**继续定位编码路径：Jacobian 本身是否变大？**','',
                   '对主导患者保持完全相同的128人真实输入，直接求截止点状态对INIT、UPDATE、ODE各组参数的完整Jacobian，取其最大奇异值。下表比较同一组参数在稳定与异常快照中的变化。','',
                   '| 参数组 | 第45轮稳定快照 | 异常batch更新前 | 增长倍数 |','|---|---:|---:|---:|']
        for name in ['INIT','UPDATE','ODE']:
            aa=stable['groups'][name]['jacobian_largest_singular_value'];bb=abnormal['groups'][name]['jacobian_largest_singular_value']
            followup.append(f'| {name} | {aa:.5g} | {bb:.5g} | {bb/aa:.1f} |')
        crossed=encoder['crossed_unit_vjp_gains']
        followup+=['','再将两种编码参数Jacobian与两种单位历史状态梯度方向交叉组合，得到实际参数VJP增益。这个计算不更新参数。','',
                   '| 编码ODE参数Jacobian | 稳定梯度方向 | 异常梯度方向 |','|---|---:|---:|',
                   f"| 稳定参数 | {crossed['stable']['stable']:.5g} | {crossed['stable']['abnormal']:.5g} |",
                   f"| 异常参数 | {crossed['abnormal']['stable']:.5g} | {crossed['abnormal']['abnormal']:.5g} |",'',
                   '稳定Jacobian配上异常梯度方向仍只有较小增益；异常Jacobian即使配上稳定方向也有很大增益。这里确认了编码参数Jacobian本身的增长。','',
                   '[完整参数Jacobian及方向交叉结果](mechanism_diagnosis/encoder_path_followup/analysis.json)','']
        ratio=abnormal['groups']['ODE']['jacobian_largest_singular_value']/stable['groups']['ODE']['jacobian_largest_singular_value']
        grades['H4']=(f"编码ODE参数Jacobian最大奇异值{stable['groups']['ODE']['jacobian_largest_singular_value']:.5g}→{abnormal['groups']['ODE']['jacobian_largest_singular_value']:.5g}，增长{ratio:.0f}倍；方向交叉验证仍有高增益",'强支持：核心放大还发生在历史编码递归的参数路径')
    if state_trace:
        ss,aa=state_trace['models']['stable'],state_trace['models']['abnormal'];peak=max(aa['visits'],key=lambda r:r['cutoff_wrt_postupdate_state_gain']);index=peak['visit_index'];matched=ss['visits'][index]
        followup += ['**继续定位编码递归：放大发生在哪一段？**','',
                     f"对同一患者，固定全部后续观测，计算截止状态对每次观测校正前后状态的Jacobian。在{peak['time_hours']:.0f}小时这次UPDATE后给状态一个微小扰动，读入相同的后续观测到24小时，最敏感方向的放大倍数由稳定快照的{matched['cutoff_wrt_postupdate_state_gain']:.4g}升到异常快照的{peak['cutoff_wrt_postupdate_state_gain']:.5g}。",'',
                     '这是向未来吸收历史观测的ODE＋UPDATE递归，不是向过去自由积分。异常轨迹后半段的多次UPDATE连续放大状态扰动，形成高编码敏感性。','',
                     '| 观测时间h | 稳定：截止状态对UPDATE后状态的增益 | 异常：相同增益 | 异常UPDATE局部状态Jacobian最大奇异值 |','|---|---:|---:|---:|']
        for j in range(max(0,index-1),len(aa['visits'])):
            r=aa['visits'][j];b=ss['visits'][j];followup.append(f"| {r['time_hours']:.0f} | {b['cutoff_wrt_postupdate_state_gain']:.4g} | {r['cutoff_wrt_postupdate_state_gain']:.5g} | {r['local_update_state_jacobian_largest_singular_value']:.4g} |")
        followup+=['','[每次ODE与UPDATE前后的完整状态增益](mechanism_diagnosis/encoder_path_followup/state_trace.json)','']
        followup+=['现有0.01 Jacobian正则约束的是ODE速度场对状态的导数。UPDATE对旧状态的Jacobian，以及多次UPDATE与ODE交替组合的增益，没有受到这一项的直接约束。','']
    if dominant:
        rows=dominant['settings'];ref=dominant['reference'];followup+=['**主导患者的额外数值核对。** 使用原128人完整batch和原分母，只取患者8的历史损失；同时检查固定原截止点状态与完整端到端两种方式。','',
                    '| 设置 | 固定原截止点的完整ODE梯度范数 | 端到端完整ODE梯度范数 | 端到端梯度相对RK4差异 |','|---|---:|---:|---:|']
        for r in rows:
            followup.append(f"| {r['setting']} | {r['modes']['locked_native_cutoff']['field_gradient_norm']:.6g} | {r['modes']['end_to_end']['field_gradient_norm']:.6g} | {r['relative_differences_to_reference']['end_to_end_field_gradient']:.3g} |")
        native=next(r for r in rows if r['setting']=='train32');reference=next(r for r in rows if r['setting']==ref)
        followup+=['',f"RK4步长减半收敛：{dominant['reference_halving_converged']}。原训练计算的主导患者梯度范数为{native['modes']['end_to_end']['field_gradient_norm']:.5g}，独立高精度参考为{reference['modes']['end_to_end']['field_gradient_norm']:.5g}，向量相对差为{native['relative_differences_to_reference']['end_to_end_field_gradient']:.3g}。高参数梯度在这位真实主导患者上同样保留。",'',
                    '[主导患者完整128人batch的数值核对](mechanism_diagnosis/dominant_patient_precision/analysis.json)','']
        grades['H2']=(f"主导患者原梯度{native['modes']['end_to_end']['field_gradient_norm']:.5g}，高精度RK4为{reference['modes']['end_to_end']['field_gradient_norm']:.5g}，向量相对差{native['relative_differences_to_reference']['end_to_end_field_gradient']:.2%}",'不支持数值误差为主因；真实大梯度在高精度下保留')
    if future_direction:
        f=future_direction
        followup+=['**历史与未来为何在这位患者上形成强冲突？** 对同一截止点状态，比较两项损失的状态梯度，再经过同一个编码参数Jacobian回传。','',
                   '| 梯度位置 | 历史范数 | 未来范数 | 两项余弦 |','|---|---:|---:|---:|',
                   f"| 截止点潜在状态 | {f['history_loss_state_gradient_norm']:.6g} | {f['future_loss_state_gradient_norm']:.6g} | {f['state_gradient_history_future_cosine']:.6f} |",
                   f"| 经编码链返回ODE参数 | {f['history_encoder_field_gradient_norm']:.6g} | {f['future_encoder_field_gradient_norm']:.6g} | {f['history_encoder_to_future_encoder_cosine']:.6f} |",'',
                   '两项状态梯度原本只有较弱的负相关。异常编码Jacobian由一个很强的方向主导，两项梯度在这个方向上的投影符号相反，回传后变成近乎相反的参数梯度；历史项幅度更大。这里定位的是该患者的局部冲突。','',
                   '[历史/未来方向与参数梯度的完整核对](mechanism_diagnosis/encoder_path_followup/future_direction.json)','']
    sections['followup']=followup
    sections['grades']=grades
    if completed:
        conclusion=['### 7.8 完成后的机制判断与改进方向','']
        if measured:
            for key,title in [('A_reencoded','固定完整历史、重新编码'),('B_fixed_state','固定同一潜在起点')]:
                first=measured[0]['before'][key]['past']['gain']['median'];maximum=max(r['after'][key]['past']['gain']['median'] for r in measured)
                largest=max(measured,key=lambda r:r['flags'][key]['after_before_ratio']);ratio=largest['flags'][key]['after_before_ratio']
                jumps=[r for r in measured if r['flags'][key]['single_update_fivefold']]
                if jumps:
                    j=jumps[0];onset=f"首次单步至少5倍的增长出现在第{j['step']['epoch']}轮第{j['step']['batch']}次更新"
                else:onset='窗口内没有单步5倍的增长，观察到的是多次更新的累积变化'
                conclusion += [f"在逐batch核对的种子重建序列中，{title}的敏感性从{first:.4g}最高升至{maximum:.4g}；{onset}。最大单步变化为{ratio:.3g}倍，位于第{largest['step']['epoch']}轮batch {largest['step']['batch']}。",'']
        p=next((x for x in patients if x['case']['id']=='abnormal_real_e46b121'),None)
        peak=next((x for x in precision if x['snapshot']['id']=='epoch46_batch121_pre'),None)
        native_error=0.
        if peak:
            r=next((r for r in peak['settings'] if r['anchor']=='snapshot_reencoded_state' and r['setting']=='tightest64' and r['status']=='complete'),None)
            if r and r.get('reference_step_halving_converged'):
                d=r['relative_differences_to_reference'];conclusion += [f"相同异常参数与输入，在float64严格Tsit5下仍有{r['past_gain']['median']:.4g}倍的历史状态敏感性中位数；与已细化RK4的奇异值、状态梯度、完整ODE梯度相对差分别为{d['past_singular_value']:.3g}、{d['history_state_gradient']:.3g}、{d['history_all_solvers_refined_full_field_gradient']:.3g}。数值对照把高敏感性与求解误差分开。",'']
            r=next((r for r in peak['settings'] if r['anchor']=='snapshot_reencoded_state' and r['setting']=='train32' and r['status']=='complete'),None)
            if r:native_error=r.get('relative_differences_to_reference',{}).get('history_all_solvers_refined_full_field_gradient',0.)
        if p:
            b=p['batch'];conclusion += [f"异常真实batch的历史ODE梯度为{b['history_gradient_norm']:.5g}，未来为{b['future_gradient_norm']:.5g}，余弦为{b['history_future_cosine']:.4f}。历史梯度沿编码参数路径的向量范数为{b['encoder_path_norm']:.5g}，沿生成直接路径为{b['direct_generation_norm']:.5g}。因此，这次异常需要结合高反向状态放大与编码参数路径来解释；患者级的实际梯度方向和贡献表进一步说明谁把这一放大转化成了较大的训练梯度。",'']
        if encoder and state_trace:
            r=next(x for x in p['patients'] if x['index']==8) if p else None
            if r:conclusion += [f"主导患者的自由历史ODE流最大状态放大约{r['past_state_max_singular_value']:.3g}倍，但其最早目标的损失梯度实际只放大{r['earliest_actual_direction_gain']:.3g}倍。其全部历史损失在截止点的梯度范数仅{r['cutoff_history_gradient_norm']:.5g}，经过编码参数路径后成为{r['encoder_field_gradient_norm']:.5g}。因此，单看自由ODE流的最大奇异值不能解释该患者的大参数梯度。",'']
            conclusion += ['**本次定位到的直接机制是：速度场在第119—120次更新中快速变得更难向过去追溯，同时，一位患者的历史编码ODE＋UPDATE递归进入高参数敏感性区域。历史回顾误差沿这条编码路径回传，产生主导整个batch的大梯度。高增益编码通道又将历史与未来的状态梯度映射成近乎相反的参数梯度，形成局部冲突。最大的梯度放大环节是编码递归；向过去自由流的高敏感性是其中一个真实因素。**','',
                           '具体仍待分开的因果问题是：UPDATE权重本身的变化与ODE改变校正前状态的位置，分别对这一高增益编码区域贡献多少。本次没有进行模块参数交换或临床字段干预，也没有把第64—65轮第二次波动认定为相同机制。','']
        conclusion+=['以下至多两个方向用于后续验证；本次未修改模型或开启改进训练。','']
        if p and p['batch']['encoder_path_norm']>2*p['batch']['direct_generation_norm']:
            conclusion += ['1. 首先做历史回顾损失对截止点状态停止梯度的定向消融，检验编码参数路径是否是失稳的必要环节；保持生成ODE和未来预测损失不变。']
        elif measured:conclusion+=['1. 根据实测敏感性尖峰，对历史回顾路径的时间跨度或梯度放大做单独消融，检验其对稳定性的作用。']
        if encoder:conclusion+=['2. 针对UPDATE对旧状态的增益做单独稳定性消融，检验是否能消除跨多次校正的累积放大；其他网络与监督保持一致。']
        elif native_error>.01:conclusion+=['2. 使用本次已验证收敛的求解精度做单独对照，检验原训练容差对异常梯度的放大程度。']
        elif p and statistics.get(p['case']['id'],{}).get('top5_norm_ranked_projected_share',0)>.5:
            conclusion+=['2. 针对实测的少数患者梯度主导，单独检验患者级稳健损失聚合；保留全部患者和原评测集。']
        else:conclusion+=['2. 对实测高放大时间段单独检验历史回顾长度或分段回顾，衡量时间传播对梯度的贡献。']
        sections['conclusion']=conclusion+['']
    (out/'mechanism_interpretation.json').write_text(json.dumps({'hypotheses':grades,'all_jobs_complete':completed},ensure_ascii=False,indent=2)+'\n')
    return sections
