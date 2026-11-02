import os
os.environ.pop("HF_HUB_OFFLINE", None)
from lm_eval.tasks import TaskManager, get_task_dict
d = get_task_dict(["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande"], TaskManager())
for k, t in d.items(): print(k, len(list(t.eval_docs)))
print("PREFETCH_OK")
