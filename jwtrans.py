# ------------------------------------------------------------------------
#
# Jordan-Wigner transformation maps fermion problem into spin problem
# 
# |1> => |alpha> and |0> => |beta >:
#
#    a_j^+ => Prod_{l=1}^{j-1}(-sigma_z[l]) * sigma_+[j]
#    a_j   => Prod_{l=1}^{j-1}(-sigma_z[l]) * sigma_-[j] 
#
# ------------------------------------------------------------------------
#
# Alternatively, we can use another convention [[[ Used ]]]
#
# |0> => |alpha> and |1> => |beta >: 
#
#    a_j^+ => Prod_{l=1}^{j-1}(sigma_z[l]) * sigma_-[j]
#    a_j   => Prod_{l=1}^{j-1}(sigma_z[l]) * sigma_+[j] 
#
# ------------------------------------------------------------------------
import numpy

# Identity
sigma_0 = numpy.identity(2, dtype=complex)
# Sigma_x
sigma_x = numpy.array([[0, 1], [1, 0]], dtype=complex)
# Sigma_y
sigma_y = numpy.array([[0, -1.j], [1.j, 0]], dtype=complex)
# Sigma_z
sigma_z = numpy.array([[1, 0], [0, -1]], dtype=complex)
# Sigma_+/-
sigma_p = 0.5 * (sigma_x + 1.j * sigma_y)
sigma_m = 0.5 * (sigma_x - 1.j * sigma_y)
# Spin_a/b
spin_a = numpy.array([1.0, 0.0])
spin_b = numpy.array([0.0, 1.0])

# ==========================
# Second convention (used)
# ==========================
# cre = sigma_m.real
# ann = sigma_p.real
# sgn = sigma_z.real
# idn = sigma_0.real
# nii = cre.dot(ann)
sgn = numpy.array([[1., 0.], [0., -1.]])
idn = numpy.array([[1., 0.], [0., 1.]])
idnt = numpy.array([[1., 0.], [0., -1.]])
cre = numpy.array([[0., 0.], [1., 0.]])
cret = numpy.array([[0., 0.], [1., 0.]])
ann = numpy.array([[0., 1.], [0., 0.]])
annt = numpy.array([[0., -1.], [0., 0.]])
nii = numpy.array([[0., 0.], [0., 1.]])
niit = numpy.array([[0., 0.], [0., -1.]])

if __name__ == '__main__':
    # test
    print(sigma_p)
    print(sigma_m)
    print(spin_a)
    print(sigma_p @ spin_a)
    print(sigma_p @ spin_b)
    print(numpy.kron(spin_a, spin_a))
    print(numpy.kron(spin_a, spin_b))

    # Matrix representation of operators in a direct product basis
    print('[a1]_{12}')
    a1 = numpy.kron(sigma_p, sigma_0)
    print(a1.real)
    print('[a2]_{12}')
    # This is because we choose |alpha>=|1>, so exchange 1 time
    a2 = numpy.kron(-sigma_z, sigma_p)
    print(a2.real)
    print("Check {a1^+,a2^+}=0")
    print((a1 @ a2).real)
    print((a2 @ a1).real)
    print(a1 @ a2 + a2 @ a1)
    print("(-sz)(-sz)")
    print(numpy.kron(-sigma_z, -sigma_z).real)
