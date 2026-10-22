"""Official four-shot MBPP demonstrations used by this campaign."""

def list_fewshot_samples():
    from lm_eval.tasks.mbpp.utils import list_fewshot_samples as canonical
    samples = list(canonical())
    samples.append({
        "task_id": 5,
        "text": "Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.",
        "code": "def count_ways(n): \r\n\tA = [0] * (n + 1) \r\n\tB = [0] * (n + 1) \r\n\tA[0] = 1\r\n\tA[1] = 0\r\n\tB[0] = 0\r\n\tB[1] = 1\r\n\tfor i in range(2, n+1): \r\n\t\tA[i] = A[i - 2] + 2 * B[i - 1] \r\n\t\tB[i] = A[i - 1] + B[i - 2] \r\n\treturn A[n] ",
        "test_list": [
            "assert count_ways(2) == 3",
            "assert count_ways(8) == 153",
            "assert count_ways(12) == 2131",
        ],
        "is_fewshot": True,
    })
    return samples
