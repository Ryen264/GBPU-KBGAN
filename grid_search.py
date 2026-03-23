import subprocess
import itertools
import json

# Grid các hyperparameter muốn tune
temperature_list = [0.5, 1.0, 2.0]
n_sample_list = [10, 20, 30]
class_rank_balance_list = [0.2, 0.5, 0.8]

results = []

for temperature, n_sample, class_rank_balance in itertools.product(temperature_list, n_sample_list, class_rank_balance_list):
    # Tạo command chạy main.py với mode gan-train và các hyperparameter
    cmd = [
        "python", "main.py",
        "mode=gan-train",
        f"--KBGAN.temperature={temperature}",
        f"--KBGAN.n_sample={n_sample}",
        f"--KBGAN.class_rank_balance={class_rank_balance}"
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd)
    output = completed.stdout

    # Trích xuất best validation performance từ output (giả sử có dòng này)
    # best_perf = None
    # for line in output.splitlines():
    #     if "Best validation performance while training:" in line:
    #         best_perf = line.split(":")[-1].strip()
    #         break

    # results.append({
    #     "temperature": temperature,
    #     "n_sample": n_sample,
    #     "class_rank_balance": class_rank_balance,
    #     "best_perf": best_perf
    # })

# Lưu kết quả
# with open("grid_results_gan_train.json", "w") as f:
#     json.dump(results, f, indent=4)