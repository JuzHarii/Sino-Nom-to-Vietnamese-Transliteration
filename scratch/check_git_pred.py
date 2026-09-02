import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Let's check git show on hcmus_predictions.txt
raw_pred = subprocess.check_output(
    ['git', 'show', '2473960d9587c93270077afe4477f12362645f72:data/result/hcmus_predictions.txt'],
    encoding='utf-8'
)
lines = raw_pred.strip().split('\n')
print(f"git hcmus_predictions.txt: {len(lines)} lines")
for line in lines[:5]:
    print(" ", line)
