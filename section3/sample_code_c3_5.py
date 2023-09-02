# numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

# 行列を定義
A = np.matrix([[3., 1., 2.], [2., 3., 1.]])


# ベクトルを定義
b = np.array([2., 1., 4.])


# 行列を表示
print("A= \n", A)
print("shape(A) = ", np.shape(A))

# ベクトルを表示
print("b= \n", b)
print("shape(b) = ", np.shape(b))

print("")
# 行列Aにベクトルbをかける
print("Ab= \n", np.dot(A, b))
print("shape(Ab) = ", np.shape(np.dot(A, b)))
