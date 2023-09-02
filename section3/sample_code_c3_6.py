# numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

# 行列を定義
a = np.matrix([3. + 2.j, 1. - 1.j, 2. + 2.j])
print("a = ", a)
print("shape(a) = ", np.shape(a))

print("")
# ベクトルを定義
b = np.array([2. + 5.j, 1. - 1.j, 4. + 1.j])
print("b = ", b)
print("shape(b) = ", np.shape(b))

print("")
# ベクトルの内積計算
print("a^Hb=", np.inner(np.conjugate(a), b))
print("shape(a^Hb) = ", np.shape(np.inner(np.conjugate(a), b)))

print("")
# ベクトルの内積計算
print("a^Ha=", np.inner(np.conjugate(a), a))
print("shape(a^Ha) = ", np.shape(np.inner(np.conjugate(a), a)))
