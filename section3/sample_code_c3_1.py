# numpyをインポート（行列を扱う各種関数を含む）
import numpy as np

# 行列を定義
A = np.matrix([[3., 1., 2.], [2., 3., 1.]])

# 行列の大きさを表示
print("行列Aの大きさ(M行, N列): ", np.shape(A))

# 行列を表示
print("A= \n", A)

print("")
# 行列Aにスカラーcをかける
c = 2.
print("cA= \n", c * A)

print("")
# 行列Aに行列Bを足す
B = np.matrix([[-1., 2., 4.], [1., 8., 6.]])
print("A+B= \n", A + B)
print("-----")

print("")
# 行列Aに行列Bをかける
B = np.matrix([[4., 2.], [-1., 3.], [1., 5.]])

print("A =", A)
print("A.shape =", A.shape)
print("-----")
print("B =", B)
print("B.shape =", B.shape)

print("-----")
print("AB [行列積] = \n", np.matmul(A, B))
print("-----")

print("AB [アインシュタインの縮約記法 行列積] = \n", np.einsum("mk,kn->mn", A, B))
print("-----")

print("")
# 行列Aと行列Bのアダマール積
B = np.matrix([[-1., 2., 4.], [1., 8., 6.]])
print("A =", A)
print("A.shape =", A.shape)
print("-----")
print("B =", B)
print("B.shape =", B.shape)
print("-----")
print("A*B [アダマール積] = \n", np.multiply(A, B))
print("-----")

print("")
# 行列Aの転置
print("A =", A)
print("A.shape =", A.shape)
print("-----")
print("A^T= \n", A.T)
print("A^T= \n", np.transpose(A, axes=(1, 0)))
print("A^T= \n", np.swapaxes(A, 1, 0))

print("")
# 複素行列のエルミート転置
A = np.matrix([[3., 1. + 2.j, 2. + 3.j], [2., 3. - 4.j, 1. + 3.j]])
print("A =", A)
print("A.shape =", A.shape)
print("-----")
print("A^H [エルミート転置] = \n", A.H)
print("A^H [エルミート転置] = \n", np.swapaxes(np.conjugate(A), 1, 0))

print("")
# 行列の積に対するエルミート転置
A = np.matrix([[3., 1. + 2.j, 2. + 3.j], [2., 3. - 4.j, 1. + 3.j]])
B = np.matrix([[4. + 4.j, 2. - 3.j], [-1. + 1.j, 3. - 2.j], [1. + 3.j, 5. + 5.j]])
print("A =", A)
print("A.shape =", A.shape)
print("-----")
print("B =", B)
print("B.shape =", B.shape)
print("-----")
print("AB [行列積] = \n", np.matmul(A, B))
print("(AB)^H [行列積に対するエルミート転置] = \n", (np.matmul(A, B)).H)
print("B^H A^H [行列積に対するエルミート転置] = \n", np.matmul(B.H, A.H))

print("")
# 単位行列の定義
matrix_I = np.eye(N=3)
print("I = \n", matrix_I)
