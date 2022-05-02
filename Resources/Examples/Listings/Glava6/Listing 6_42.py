# Listing 6.42
# Параметры уровня 1
W1 = model.get_weights()[0]
b1 = model.get_weights()[1]
# Параметры уровня 2
W2 = model.get_weights()[2]
b2 = model.get_weights()[3]

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
