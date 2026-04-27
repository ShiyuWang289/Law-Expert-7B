import matplotlib.pyplot as plt

models = ["Base","SFT","DPO"]
general = [78.83,79.42,79.05]
legal = [9.9,13.3,16.7]  

plt.figure(figsize=(7,4))
plt.plot(models, general, marker='o', label="General (ceval avg)")
# 仅画有值点
legal_x = [m for m,v in zip(models, legal) if v is not None]
legal_y = [v for v in legal if v is not None]
plt.plot(legal_x, legal_y, marker='s', label="Legal coverage")

plt.title("Capability Trade-off (General vs Legal)")
plt.ylabel("Score")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("analysis/p15_tradeoff.png", dpi=180)
print("saved: analysis/p15_tradeoff.png")