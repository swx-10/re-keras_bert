import matplotlib.pyplot as plt

passed = 0
steps = 30
learning_rate = 5e-5
min_learning_rate = 1e-5

x,y=[],[]

for i in range(100):
    if passed < steps:
        lr = (passed + 1.) / steps * learning_rate
        y.append(lr)
        x.append(passed)
        passed += 1
        
    elif steps <= passed < steps * 2:
        lr = (2 - (passed + 1.) / steps) * (learning_rate - min_learning_rate)
        lr += min_learning_rate
        y.append(lr)
        x.append(passed)
        passed += 1

plt.plot(x,y)
plt.show()