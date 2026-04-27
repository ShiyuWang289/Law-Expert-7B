import json, os

runs = {
    "0.05": "/root/autodl-tmp/saves/dpo_beta_005",
    "0.1":  "/root/autodl-tmp/saves/dpo_beta_01",
    "0.2":  "/root/autodl-tmp/saves/dpo_beta_02",
}

print("beta,train_loss,eval_loss,margin,acc")
for b, d in runs.items():
    p = os.path.join(d, "all_results.json")
    if not os.path.exists(p):
        print(f"{b},NA,NA,NA,NA")
        continue
    x = json.load(open(p, "r", encoding="utf-8"))
    print(f"{b},{x.get('train_loss')},{x.get('eval_loss')},{x.get('eval_rewards/margins')},{x.get('eval_rewards/accuracies')}")