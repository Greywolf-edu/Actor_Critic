G = 0
I = 1
t = 0
T = 5
Rewards = [0] + [1 for i in range(T)]
for k in range(t + 1, T + 1):  # k = 1, 2, 3, 4, 5
    G += I * Rewards[k]
    I *= 0.8

print(Rewards)
print(G)
print(1+0.8 + 0.8**2 + 0.8**3 + 0.8**4)