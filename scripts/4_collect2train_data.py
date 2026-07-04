import os
import subprocess

root_dir = "/home/dobot/projects/datasets/"
dataset_name = "dataset_package_test"

dataset_dir = os.path.join(root_dir, dataset_name, "collect_data")
all_entries = os.listdir(dataset_dir)
all_data_dir = [d for d in all_entries if os.path.isdir(os.path.join(dataset_dir, d))]
all_data_dir.sort(key=lambda x: int(x))
print(all_data_dir)

idx = 0
for i in all_data_dir:
    print("dealing with: ", i)
    CMD = [
        'python', 'script_collect2train.py',
        '--root_dir', root_dir,
        '--dataset_name', dataset_name,
        '--date_collect', i,
        '--idx', str(idx)
    ]
    rt_code = subprocess.run(CMD).returncode
    if rt_code:
        break
    idx += 1
