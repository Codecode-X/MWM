"""Observable deterministic interventions, reusable by model rollout code."""
POLICY_NAMES = ['Never Treat', 'Always Treat', 'Delayed Treatment',
                'Aggressive Dynamic', 'Conservative Dynamic']


def decide(policy_id, time, fork_time, current_X2, specs):
    if policy_id == 0:
        return 0
    if policy_id == 1:
        return 1
    if policy_id == 2:
        return int(time >= fork_time + specs['delay_years'])
    if policy_id in (3, 4):
        return int(current_X2 > specs['thresholds'][str(policy_id)])
    raise ValueError(f'unknown policy {policy_id}')


def treatment_duration(intervals, start, end):
    return sum(max(0.0, min(b, end) - max(a, start)) * u for a, b, u in intervals)
