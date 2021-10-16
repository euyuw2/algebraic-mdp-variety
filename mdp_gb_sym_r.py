# from sympy import ones, symbols, det, nsimplify
import sympy as sp
from sympy.abc import i, j, k
import numpy as np
import time
from sage.all import Macaulay2


# from sage.all import *
# def


def mdp_manual():
    # num_states = 2
    # num_actions = 3
    # r = np.array([[-0.1, -1, 0.1], [0.4, 1.5, 0.1]])  # |S| * |A|
    # P = np.zeros([num_states, num_actions, num_states])
    #
    # P[0] = [[0.9, 0.1],  # s1, a1
    #         [0.2, 0.8],
    #         [0.7, 0.3]]  # s1, a3
    # P[1] = [[0.05, 0.95],  # s2, a1
    #         [0.25, 0.75],
    #         [0.3, 0.7]]  # s2, a3

    num_states = 2
    num_actions = 2
    r = np.array([[-1, -1], [1, 1]])  # |S| * |A|
    P = np.zeros([num_states, num_actions, num_states])

    P[0] = [[0.96, 0.04],  # s1, a1
            [0.19, 0.81]]  # s1, a2
    P[1] = [[0.43, 0.57],  # s2, a1
            [0.72, 0.28]]  # s2, a2
    return P, r


def random_prob_vec(N, decimal=2):
    prob = np.zeros([N])
    perm = np.random.permutation(N)
    scalar = 1
    for i in range(N):
        if i == N - 1:
            prob[perm[i]] = scalar
            return prob
        p = np.around(np.random.random() * scalar, decimal)
        prob[perm[i]] = p
        scalar = 1 - np.sum(prob)


def mdp_gen_NsNa(num_states, num_actions, save=False, decimal=2):
    # reward = np.around(np.random.random([num_states, num_actions]) * 2 - 1, decimal)
    reward = np.random.randint(0, 10, [num_states, num_actions]) - 5
    transition = np.zeros([num_states, num_actions, num_states])
    for i in range(num_states):
        for j in range(num_actions):
            prob_vec = random_prob_vec(num_states, decimal)
            transition[i, j, :] = prob_vec
    if save:
        save_mdp(reward=reward, transition=transition, save_path='saved_mdp/')
        print('MDP saved')
    transition = np.around(transition, 2)
    return transition, reward


def save_mdp(reward, transition, save_path):
    num_states, num_actions = reward.shape[0], reward.shape[1]
    np.save(save_path + 'ns' + str(num_states) + '_na' + str(num_actions) + '_reward.npy', reward)
    np.save(save_path + 'ns' + str(num_states) + '_na' + str(num_actions) + '_transition.npy', transition)


def gen_r_pi_expr(r, pi):
    num_states, num_actions = pi.shape[0], pi.shape[1]
    r_pi = sp.ones(num_states, 1)
    for i in range(num_states):
        sum_expr = None
        for j in range(num_actions):
            if sum_expr is None:
                sum_expr = r[i][j] * pi[i, j]
            else:
                sum_expr += r[i][j] * pi[i, j]
        r_pi[i] = sum_expr
    return r_pi


def gen_P_pi_expr(P, pi):
    num_states, num_actions = pi.shape[0], pi.shape[1]
    P_pi = sp.ones(num_states, num_states)
    for j in range(num_states):
        for k in range(num_states):
            sum_expr = None
            for i in range(num_actions):
                if sum_expr is None:
                    # sum_expr = P[j, i, k] * pi[j, i]
                    sum_expr = P[j][i][k] * pi[j, i]
                else:
                    sum_expr += P[j][i][k] * pi[j, i]
            P_pi[j, k] = sum_expr
    return P_pi


def get_Ai(A, i, r):
    A_temp = A[:, :]
    A_temp.col_del(i)
    # r_insert = Matrix([i for i in r])
    A_temp = A_temp.col_insert(i, r)
    return A_temp


def gen_mdp_symbols(num_states, num_actions):
    T = []
    R = []
    for s in range(num_states):
        p_state = []
        r_state = []
        for a in range(num_actions):
            p_action = []
            for s_ in range(num_states):
                p_action.append(sp.symbols('T' + str(s) + str(a) + str(s_)))
            p_state.append(p_action)
            r_state.append(sp.symbols('R' + str(s) + str(a)))
        R.append(r_state)
        T.append(p_state)
    return T, R


def gen_symbols(num_states, num_actions):
    p = []
    for s in range(num_states):
        p_state = []
        for a in range(num_actions):
            p_state.append(sp.symbols('p' + str(s) + str(a)))
        p.append(p_state)
    v = []
    for s in range(num_states):
        v.append(sp.symbols('v' + str(s)))
    return p, v


# def calc_grobner()

def gen_pi_expr(pi):
    num_states, num_actions = pi.shape
    expr_list = []
    for i in range(num_states):
        sum_expr = None
        for j in range(num_actions):
            if sum_expr is None:
                sum_expr = pi[i, j]
            else:
                sum_expr += pi[i, j]
            expr_list.append(str(pi[i, j]) + '**2 - ' + str(pi[i, j]))
        expr_list.append(str(sum_expr - 1))
    return expr_list


def gen_expr(A, r_pi, v, pi):
    expr_list = []

    # cramer rule equations
    det_A_expr = sp.det(A)
    for i in range(A.shape[0]):
        A_i = get_Ai(A, i, r_pi)
        expr_list.append(str(det_A_expr * v[i] - sp.det(A_i)))
        # expr_list.append(str(sp.nsimplify(det_A_expr * v[i] - sp.det(A_i), rational=True)))
    c = sp.symbols('c')
    expr_list.append(str(det_A_expr * c - 1))
    # expr_list.append(str(sp.nsimplify(det_A_expr * c - 1, rational=True)))

    expr_prob_list = gen_pi_expr(pi)

    return expr_list + expr_prob_list


def write_exprs_to_file(fname, exprs):
    with open(fname, 'w') as f:
        for expr in exprs:
            f.write(expr + ',\n')


def create_var_list(r, pi, v):
    ns, na = pi.shape[0], pi.shape[1]
    var_list = []
    for i in range(ns):
        for j in range(na):
            var_list.append(str(pi[i, j]))

    var_list.append('c')

    for i in range(ns):
        for j in range(na):
            var_list.append(str(r[i][j]))

    for i in range(ns):
        var_list.append(str(v[i]))

    return var_list


def create_var_list_str(r, pi, v):
    ns, na = pi.shape[0], pi.shape[1]
    var_list_str = '['
    for i in range(ns):
        for j in range(na):
            var_list_str += str(pi[i, j]) + ', '

    var_list_str += 'c, '

    for i in range(ns):
        for j in range(na):
            var_list_str += str(r[i][j]) + ', '

    for i in range(ns):
        var_list_str += str(v[i]) + ', '
    var_list_str = var_list_str[:-2] + ']'

    m_order = '{' + str(ns * na + 1 + ns * na + ns) + ':1}'
    # m_order = 'Eliminate ' + str(ns * na + 1)
    # m_order = 'Lex'

    return var_list_str, m_order
    # var_list_str


def gb_macaulay2(var_list, m_order, exprs, idx_subring, path_m2='/Applications/Macaulay2-1.15/bin/M2'):
    m2 = Macaulay2(command=path_m2)
    R = m2.ring('QQ', var_list, order=m_order)
    curve = m2.ideal((exprs))
    # gb = m2.selectInSubring(idx_subring, curve.gb().gens())
    gb = curve.gb().gens()
    # print(gb)
    # gb_list = str(gb).split(' ')
    gb_list = str(gb).split(' ')[1:-1]
    return gb_list


def mdp_gb_sym_r(P, r_sym, gamma, fname='expr_temp.txt'):
    ns, na = len(P), len(P[0])
    pi, v = gen_symbols(ns, na)
    pi = sp.Matrix(pi)

    P = sp.nsimplify(P, rational=True)
    # P = sp.nsimplify(P, tolerance=1e-1)
    # r = sp.nsimplify(r, rational=True)
    gamma = sp.nsimplify(gamma, rational=True)

    P_pi = gen_P_pi_expr(P, pi)
    A = sp.eye(ns) - gamma * P_pi
    r_pi = gen_r_pi_expr(r_sym, pi)

    # A = sp.nsimplify(A, rational=True)
    # r_pi = sp.nsimplify(r_pi, rational=True)
    print('generating expressions ...')
    exprs = gen_expr(A, r_pi, v, pi)
    write_exprs_to_file(fname, exprs)
    var_list_str, m_order = create_var_list_str(r_sym, pi, v)  # here
    var_list = create_var_list(r_sym, pi, v)

    command = 'R, ' + var_list_str + ' = PolynomialRing(QQ, ' + str(var_list) + ', order=\'lex\').objgens()'

    print(command)

    # start_time = time.time()
    # gb_list = gb_macaulay2(var_list_str, 'Lex', exprs, ns * na + 1)
    #
    # for gb in gb_list:
    #     print(gb)
    #     print('===')
    # print('%.4f seconds elapsed' % (time.time() - start_time))
    # m2 = Macaulay2(command='/Applications/Macaulay2-1.15/bin/M2')
    # R2 = m2.ring('QQ', '[p11, p12, p21, p22, c, v0, v1]', order='Eliminate 5')
    # # R = m2('QQ[p11, p12, p21, p22, c, v0, v1], MonomialOrder => Lex')
    # curve = m2.ideal


def playground():
    # x, y, z, t = symbols('x y z t')

    # gb = groebner([x**4 - y**5, x**3 - y**7], order='grevlex')
    # gb = groebner([x*y - z*t], x, y, z, t, order='grevlex')
    # print(gb.exprs)
    # r, P = mdp_manual()
    g = sp.ones(2, 2)

    pi, v = gen_symbols(4, 4)
    # g[0, 0] = p11
    # v1, v2 = sp.symbols('v1, v2')
    # expr1 = p11 * 2
    # expr2 = p21 * p22
    pi = sp.Matrix(pi)
    pi = pi.inv()
    print(pi)
    # e1 = 2 * pi[0, 0]
    # print(g)

    # print(det(pi))
    # pi = MatrixSymbol('pi', 2, 2)
    # expr = det(pi)
    # g = groebner([det(pi), det(pi ** 2 + ones(2, 2))])
    # exps = g.exprs
    # for exp in exps:
    #     print(type(exp))
    # print(g.exprs)
    # print(g)

    # print(expr.subs({pi: r}).doit())
    # r = Matrix(r)
    # expr = Sum(pi[i, j], (i, 0, 1), (j, 0, 1))
    # print(expr)
    # print(groebner(expr, *pi, order='grevlex'))
    # r_pi = r.multiply_elementwise(pi)

    # y = MatrixSymbol('y', 3, 2)
    # print(r_pi)
    # gb = groebner([])


if __name__ == '__main__':
    P_sym, r_sym = gen_mdp_symbols(2, 2)
    P, r = mdp_gen_NsNa(2, 2)
    mdp_gb_sym_r(P, r_sym, 0.9)
