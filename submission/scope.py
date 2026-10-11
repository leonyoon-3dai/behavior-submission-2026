"""Keep the preserved six-task pilot separate from the final five-task submission."""
PILOT_TASK_IDS = (0, 1, 2, 3, 5, 7)
SUBMISSION_TASK_IDS = (0, 1, 2, 3, 5)
PUBLIC_INDICES = tuple(range(4))
EVALUATION_TASK_IDS = (1, 2, 5)
SUBMISSION_ROLLOUTS = len(EVALUATION_TASK_IDS) * len(PUBLIC_INDICES)
EVALUATION_TASK_ORDER = (2, 1, 5)

def validate_final_manifest(manifest):
    expected = set(SUBMISSION_TASK_IDS)
    if set(manifest['supported_task_ids']) != expected or set(map(int, manifest['routes'])) != expected:
        raise ValueError('Final submission requires exactly tasks 000,001,002,003,005; task007 is pilot only')
    if any(route['condition'] != 'RT-EXPO' for route in manifest['routes'].values()):
        raise ValueError('All five final routes must use the user-confirmed A0+RT policy')
