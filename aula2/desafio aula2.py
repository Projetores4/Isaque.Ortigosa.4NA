x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

n = len (x)
W_x = sum (x)
W_y = sum (y)
W_xy = sum (xi * yi for xi, yi in zip (x, y))
W_xx = sum (xi ** 2 for xi in x)

a = (n * W_xy - W_x * W_y) / (n * W_xx - W_x ** 2)
b = (W_y - a * W_x) / n

print(f"Coeficiente angular: {a}")
print(f"Coeficiente linear: {b}")
print(f"Equação da reta: {a:.2f}x + {b:.2f}")