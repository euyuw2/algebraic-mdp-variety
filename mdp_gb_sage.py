# from sympy import ones, symbols, det, nsimplify
import sympy as sp
from sympy.abc import i, j, k
import numpy as np
import time
from sage.all import QQ
from sage.all import PolynomialRing
from sage.all import ideal
# import sage.interfaces.
# from sage.all import Macaulay2
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

    # num_states = 2
    # num_actions = 2
    # r = np.array([[-1, -1], [1, 1]]) # |S| * |A|
    # P = np.zeros([num_states, num_actions, num_states])
    #
    # P[0] = [[0.96, 0.04], # s1, a1
    #         [0.19, 0.81]] # s1, a2
    # P[1] = [[0.43, 0.57], # s2, a1
    #         [0.72, 0.28]] # s2, a2
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
    num_states, num_actions = r.shape[0], r.shape[1]
    r_pi = sp.ones(num_states, 1)
    for i in range(num_states):
        sum_expr = None
        for j in range(num_actions):
            if sum_expr is None:
                sum_expr = r[i, j] * pi[i, j]
            else:
                sum_expr += r[i, j] * pi[i, j]
        r_pi[i] = sum_expr
    return r_pi


def gen_P_pi_expr(P, pi):
    num_states, num_actions = P.shape[0], P.shape[1]
    P_pi = sp.ones(num_states, num_states)
    for j in range(num_states):
        for k in range(num_states):
            sum_expr = None
            for i in range(num_actions):
                if sum_expr is None:
                    sum_expr = P[j, i, k] * pi[j, i]
                else:
                    sum_expr += P[j, i, k] * pi[j, i]
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


def gen_symbols_pi(num_states, num_actions, sb='p'):
    p = []
    for s in range(num_states):
        p_state = []
        for a in range(num_actions):
            p_state.append(sp.symbols(sb + str(s) + str(a)))
        p.append(p_state)
    return p


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
            # expr_list.append(pi[i, j] ** 2 - pi[i, j])
            expr_list.append(str(pi[i, j]) + '**2 - ' + str(pi[i, j]))
        expr_list.append(str(sum_expr - 1))
    return expr_list


def gen_expr(A, r_pi, v, pi):
    expr_list = []

    # cramer rule equations
    det_A_expr = sp.det(A)
    for i in range(A.shape[0]):
        A_i = get_Ai(A, i, r_pi)
        expr = sp.nsimplify(det_A_expr * v[i] - sp.det(A_i), tolerance=0.1)
        expr_list.append(str(expr))
        # expr_list.append(str(sp.nsimplify(det_A_expr * v[i] - sp.det(A_i), rational=True)))
    c = sp.symbols('c')
    expr = sp.nsimplify(det_A_expr * c - 1, tolerance=0.1)
    expr_list.append(str(expr))
    # expr_list.append(str(sp.nsimplify(det_A_expr * c - 1, rational=True)))

    expr_prob_list = gen_pi_expr(pi)

    return expr_list + expr_prob_list


def write_exprs_to_file(fname, exprs):
    with open(fname, 'w') as f:
        for expr in exprs:
            f.write(expr + ',\n')


def create_var_list(pi, v):
    ns, na = pi.shape[0], pi.shape[1]
    var_list = []
    for i in range(ns):
        for j in range(na):
            var_list.append(str(pi[i, j]))
            # var_list_str += str(pi[i, j]) + ', '
    var_list.append('c')
    for i in range(ns):
        var_list.append(str(v[i]))

    return var_list


def create_var_list_str(pi, v):
    ns, na = pi.shape[0], pi.shape[1]
    var_list_str = '['
    for i in range(ns):
        for j in range(na):
            var_list_str += str(pi[i, j]) + ', '
    var_list_str += 'c, '
    for i in range(ns):
        var_list_str += str(v[i]) + ', '
    var_list_str = var_list_str[:-2] + ']'

    m_order = '{' + str(ns * na + 1 + ns) + ':1}'
    # m_order = 'Eliminate ' + str(ns * na + 1)
    # m_order = 'Lex'

    return var_list_str, m_order
    # var_list_str


# def gb_macaulay2(var_list, m_order, exprs, idx_subring, path_m2='/Applications/Macaulay2-1.15/bin/M2'):
#     m2 = Macaulay2(command=path_m2)
#     R = m2.ring('QQ', var_list, order=m_order)
#     curve = m2.ideal((exprs))
#     gb = m2.selectInSubring(idx_subring, curve.gb().gens())
#     # print(gb)
#     # gb_list = str(gb).split(' ')
#     gb_list = str(gb).split(' ')[1:-1]
#     return gb_list


def mdp_gb_exprs(P, r, gamma, fname='expr_temp.txt'):
    ns, na = r.shape[0], r.shape[1]
    pi, v = gen_symbols(ns, na)
    pi = sp.Matrix(pi)

    # P = sp.nsimplify(P, tolerance=0.1)
    # r = sp.nsimplify(r, tolerance=0.1)
    # gamma = sp.nsimplify(gamma, tolerance=0.1)

    # P = sp.nsimplify(P, rational=True)
    # r = sp.nsimplify(r, rational=True)
    # gamma = sp.nsimplify(gamma, rational=True)

    P_pi = gen_P_pi_expr(P, pi)
    A = sp.eye(ns) - gamma * P_pi
    r_pi = gen_r_pi_expr(r, pi)

    # A = sp.nsimplify(A, tolerance=1e-2)
    # r_pi = sp.nsimplify(r_pi, tolerance=1e-2)
    # r_pi = sp.nsimplify(r_pi, rational=True)
    print('generating expressions ...')
    exprs = gen_expr(A, r_pi, v, pi)
    write_exprs_to_file(fname, exprs)
    var_list_str, m_order = create_var_list_str(pi, v) # here
    var_list = create_var_list(pi, v)
    command = 'R, ' + var_list_str + ' = PolynomialRing(QQ, ' + str(var_list) + ', order=\'lex\').objgens()'
    # print(var_list_str)
    # print(var_list)
    print(command)

    # print('calculating grobner bases ... ')
    # start_time = time.time()

    # R, var_list_str = PolynomialRing(QQ, var_list_str, order='lex').objgens()
    # I = ideal(exprs)
    # R, var_list = PolynomialRing(QQ, 'xyz', order='lex')
    # print(R)
    # print(var_list)

    # print(I.groebner_basis())

    # for gb in gb_list:
    #     print(gb)
    # print('%.4f seconds elapsed' % (time.time() - start_time))
    # m2 = Macaulay2(command='/Applications/Macaulay2-1.15/bin/M2')
    # R2 = m2.ring('QQ', '[p11, p12, p21, p22, c, v0, v1]', order='Eliminate 5')
    # # R = m2('QQ[p11, p12, p21, p22, c, v0, v1], MonomialOrder => Lex')
    # curve = m2.ideal


def gen_expr_two_policies(A_pi, A_delta, r_pi, r_delta, pi, delta):
    expr_list = []

    det_A_pi_expr = sp.det(A_pi)
    det_A_delta_expr = sp.det(A_delta)
    for i in range(A_pi.shape[0]):
        A_pi_i = get_Ai(A_pi, i, r_pi)
        A_delta_i = get_Ai(A_delta, i, r_delta)
        expr = sp.nsimplify(det_A_delta_expr * sp.det(A_pi_i) -
                            det_A_pi_expr * sp.det(A_delta_i), tolerance=0.1)
        y = sp.symbols('y'+str(i))
        expr_list.append(str(expr - y**2))

    expr_list.append(str(sp.nsimplify(det_A_pi_expr * sp.symbols('c1') - 1, tolerance=0.1)))
    expr_list.append(str(sp.nsimplify(det_A_delta_expr * sp.symbols('c2') - 1, tolerance=0.1)))

    expr_list += gen_pi_expr(pi)
    expr_list += gen_pi_expr(delta)

    return expr_list


def create_var_list_two_polices(pi, delta):
    ns, na = pi.shape[0], pi.shape[1]
    var_list = []
    for i in range(ns):
        for j in range(na):
            var_list.append(str(delta[i, j]))

    var_list.append('c')
    for i in range(ns):
        for j in range(na):
            var_list.append(str(pi[i, j]))

    return var_list


def create_var_list_str_two_polices(pi, delta):
    ns, na = pi.shape[0], pi.shape[1]
    var_list_str = '['
    for i in range(ns):
        for j in range(na):
            var_list_str += str(delta[i, j]) + ', '

    var_list_str += 'c, '
    for i in range(ns):
        for j in range(na):
            var_list_str += str(pi[i, j]) + ', '

    var_list_str = var_list_str[:-2] + ']'

    return var_list_str


def mdp_gb_exprs_two_policies(P, r, gamma, fname='expr_temp.txt'):
    ns, na = r.shape[0], r.shape[1]
    pi, v = gen_symbols(ns, na)
    pi = sp.Matrix(pi)
    delta = gen_symbols_pi(ns, na, 'd')
    delta = sp.Matrix(delta)

    P_delta = gen_P_pi_expr(P, delta)
    P_pi = gen_P_pi_expr(P, pi)
    A_pi = sp.eye(ns) - gamma * P_pi
    A_delta = sp.eye(ns) - gamma * P_delta
    r_pi = gen_r_pi_expr(r, pi)
    r_delta = gen_r_pi_expr(r, delta)

    print('generating expressions ...')
    exprs = gen_expr_two_policies(A_pi, A_delta, r_pi, r_delta, pi, delta)
    write_exprs_to_file(fname, exprs)
    var_list_str = create_var_list_str_two_polices(pi, delta)
    var_list = create_var_list_two_polices(pi, delta)
    command = 'R, ' + var_list_str + ' = PolynomialRing(QQ, ' + str(var_list) + ', order=\'lex\').objgens()'
    print(command)


if __name__ == '__main__':
    # playground()

    # P, r = mdp_manual()
    P, r = mdp_gen_NsNa(2, 2, False, decimal=2)
    # mdp_gb_exprs_two_policies(P, r, 0.9)
    mdp_gb_exprs(P, r, 0.9)
    # ns2_na2(P, r, 0.9)