import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import networkx as nx
import os
from scipy.optimize import curve_fit



def fit(x, a, b):
    return a*x + b

def exponential_moving_average(x, alpha=0.1):
    fwd = np.zeros_like(x)
    bwd = np.zeros_like(x)

    fwd[0] = x[0]
    for i in range(1, len(x)):
        fwd[i] = alpha * x[i] + (1 - alpha) * fwd[i - 1]

    bwd[-1] = fwd[-1]
    for i in range(len(x) - 2, -1, -1):
        bwd[i] = alpha * fwd[i] + (1 - alpha) * bwd[i + 1]
    return bwd

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

# ---- SINGLE SIM ----

def load_single_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data
    K = data["K"]
    l0 = data["l0"]
    maxit = data["maxit"]
    epsilon = data["epsilon"]
    Nmax = data["Nmax"]
    tmax = data["tmax"]
    taus = data["tau"]
    Ns = data["N"]
    Cijs = data["Cij"]
    p0s = data["p0"]
    p1s = data["p1"]
    sample_edges = data["sample_edges"]
    sim_time = data["sim_time"]
    ls = data["ls"]
    t = np.arange(maxit)
    C0 = K/(l0**3)

def plot_probs(p0s, taus, save=False, path=None):
    p0smooth = exponential_moving_average(p0s, 0.03)
    #plt.title("Evolución de las probabilidades")
    plt.scatter(taus, p0s, marker='.', color='lightblue', alpha=0.3)
    plt.scatter(taus, 1 - p0s, marker=".", color='peachpuff', alpha=0.3)
    plt.plot(taus, p0smooth, label="Retracción")
    plt.plot(taus, 1 - p0smooth, label="Crecimiento + Bifurcación")

    plt.xlabel("T")
    plt.ylabel("Probabilidad")
    plt.legend(loc='upper right')
    if save:
        plt.savefig(path)
    plt.show()

def plot_N(Ns, taus, save=False, path=None):
    
    smoothNs = exponential_moving_average(Ns, 0.03)
    plt.scatter(taus, Ns, marker='.', color='lightblue', alpha=0.3)
    plt.plot(taus, smoothNs)
    plt.ylabel("Número de puntas")
    plt.xlabel("t")

    if save:
        plt.savefig(path)
    plt.show()

def scatter_N(Ns, taus):
    
    plt.scatter(taus, Ns, marker='.')
    #plt.scatter(taus, 2**(0.5*taus))
    plt.ylabel("Número de puntas")
    plt.xlabel("t")
    plt.savefig("graficos_inst/N_vs_tiempo_principio.png")
    plt.show()

def plot_l(ls, taus, l0, save=False, path=None):
    smoothls = exponential_moving_average(ls, 0.03)
    plt.scatter(taus, ls/l0, marker='.', color='lightblue', alpha=0.3)
    plt.plot(taus, smoothls/l0)
    plt.xlabel("t")
    plt.ylabel("Largo total")
    if save:
        plt.savefig(path)
    plt.show()

def scatter_l(ls, taus, l0):
    plt.scatter(taus, ls/l0, marker='.')
    plt.xlabel("t")
    plt.ylabel("Largo total")

    plt.savefig("graficos_inst/largo_vs_tiempo_principio.png")
    plt.show()

def plot_Cijs(Cijs, taus, C_, save=False, path=None):
    maxit = len(Cijs)

    Csmooth = exponential_moving_average(Cijs, 0.03)
    #plt.title("Cij promedio c/r al tiempo")
    plt.yscale('log')

    plt.scatter(taus, Cijs, marker='.', color='lightblue', alpha=0.3)
    plt.plot(taus, Csmooth, label="$<C_{ij}>$ de las puntas")
    plt.plot(taus, C_*np.ones(maxit), label="$\\bar{C}$")

    plt.xlabel("t")
    plt.ylabel("Conductancia")
    plt.legend()
    if save:
        plt.savefig(path)
    plt.show()

def scatter_Cijs(Cijs, taus, C_):
    maxit = len(Cijs)

    #plt.title("Cij promedio c/r al tiempo")
    plt.yscale('log')
    # plt.xscale('log')
    plt.scatter(taus, Cijs, marker='.', label="Simulación")
    #plt.plot(taus, C_*np.ones(maxit), label="$\\bar{C}$")
    #plt.plot(np.linspace(taus[0], taus[-1], 1000), 39.686/(taus**(2.09)), color="red", label="$y=39.69x^{-2.09}$")
    plt.xlabel("Iteración")
    plt.ylabel("Conductancia")
    plt.legend()
    #plt.savefig("graficos_proba/Cij_vs_tiempo")
    plt.show()   

def plot_murray(sample_edges, taus, save=False, path=None):
    means = []
    errores = []
    ts = []
    for i, sample in enumerate(sample_edges):
        puntas = get_puntas(sample)
        valores = []
        for edge in sample:
            rp = (edge[4] * edge[2])**(3/4)
            rh = 0
            for hijo in sample:
                if hijo[0] == edge[1]:
                    rh += (hijo[4] * hijo[2])**(3/4)
            if rh!=0 :
                valores.append(rp/rh)
        if len(valores) > 0:
            ts.append(taus[i])
            means.append(np.mean(valores))
            errores.append(np.std(valores)/np.sqrt(len(valores)))
    #plt.errorbar(np.arange(len(means)), means, yerr=errores, fmt='.', label="Promedio sobre un árbol")
    plt.scatter(ts, means, marker='.', label="Promedio")
    plt.plot(ts, np.ones(len(means)), label="Teoría")
    plt.xlabel("t")
    plt.ylabel("$r_p^3/(r_1^3+r_2^3)$")
    plt.ylim(0, 2)
    plt.legend()
    if save:
        plt.savefig(path)
    plt.show()

def show_graph(edge_list, puntas, n_nodes, qs, q0, it, T, K, C_, axs, animate=False,save=False, path=None):
    """
    Plotea el árbol con flujos y radios (conductancias), MUY LENTO
    """
    axs.clear()
    G = nx.DiGraph()

    G.add_nodes_from(range(n_nodes))
    
    for edge in edge_list:
        G.add_edge(edge[0], edge[1], label=f"{edge[2]:.2f}", radio=(edge[4] * edge[2])**(1/4), flow=np.sqrt(np.mean(edge[3])))

    # Color de punta dado por inflow normalizado a q0+ruido max = 2q0
    maxq = max(np.max(qs[puntas]), q0)
    node_colors = [
        'lightgrey' for n in list(G.nodes())
    ]
    # Grosor de edge dado por radio (derivado de conductancia y largo)
    radios = list(nx.get_edge_attributes(G, 'radio').values())
    max_r = np.max(radios)
    min_r = max(np.min(radios), 1e-20)
    max_width = 4
    min_width = 1
    widths = [
        (max_width - min_width)*(r-min_r)/(max_r - min_r) + min_width for r in radios
    ]

    flows = list(nx.get_edge_attributes(G, 'flow').values())
    max_flow = np.max(flows)
    min_flow = np.min(flows)
    cmap = plt.cm.viridis.reversed()
    edge_colors = [
        cmap(flow/max_flow) for flow in flows
    ]

    G.graph['graph'] = {
        'rankdir': 'TB', 
        'ranksep': '2.5',  
        'nodesep': '1.5' 
    }

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        arrows=False,
        node_size=50,
        font_size=7,
        width=widths,
        ax=axs
    )
    edge_labels = nx.get_edge_attributes(G, 'label')
    plt.title(f"T = {T:.3f}, C_={C_:.0e}, K={K}")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=axs)
    #plt.savefig(f"./arboles/{it}_{K}")

    # LEGEND

    if save:
        plt.savefig(path)
        return
    if animate:
        #plt.draw()
        plt.pause(0.000000001)
        #plt.pause(3)

    if not animate:
        plt.show()


def get_phase_change(Ns):
    idx = 0
    pers = 10
    for i in range(pers,len(Ns)):
        if all([Ns[i]<Ns[i-j] for j in range(1,pers+1)]):
            return i-pers

def get_generations(edge_list):
    sedge_list = sorted(edge_list, key=lambda edge: edge[0])
    gens = np.ones(len(edge_list))*(-1)
    for i, edge in enumerate(sedge_list):
        if edge[0] == 0:
            gens[i] = 0
            continue
        for j, padre in enumerate(sedge_list):
            if padre[1] == edge[0] and gens[j] != -1:
                gens[i] = gens[j]+1
                continue
    return gens

def get_puntas(edge_list):
    puntas = list(range(len(edge_list)+1))
    for edge in edge_list:
        if edge[0] in puntas:
            puntas.remove(edge[0])
    return puntas

def terminacion_vs_gen(edge_list):
    sedge_list = sorted(edge_list, key=lambda edge: edge[0])
    gens = get_generations(sedge_list)
    gens_short = list(set(gens))
    puntas = get_puntas(sedge_list)
    puntas_g = [0]*len(gens_short)
    for i, edge in enumerate(sedge_list):
        if edge[1] in puntas:
            puntas_g[gens_short.index(gens[i])] += 1
    for i, tot in enumerate(puntas_g):
        puntas_g[i] /= len(gens[gens==gens_short[i]])
    plt.scatter(gens_short, puntas_g)
    #plt.show()

def scatter_v_g(edge_list, v=2, save=False, path=None):
    # USAMOS LOS EDGES SORTED
    sedge_list = sorted(edge_list, key=lambda edge: edge[0])
    gens = get_generations(sedge_list)
    lengths = [(edge[2]*edge[4])**(1/4) for edge in sedge_list]
    gens_short = list(set(gens))
    av_lengths = np.zeros(len(gens_short))
    for i in range(len(lengths)):
        av_lengths[gens_short.index(gens[i])] += lengths[i]
    for i in range(len(gens_short)):
        c = len(gens[gens==gens_short[i]])
        av_lengths[i] /= c
    plt.scatter(gens_short, av_lengths)
    return gens_short, av_lengths
    #plt.show()

def scatter_v_g_t(sample_edges, v=2):
    t = []
    g = []
    l = []
    for i, edge_list in enumerate(sample_edges):
        sedge_list = sorted(edge_list, key=lambda edge: edge[0])
        gens = get_generations(sedge_list)
        lengths = [edge[v] for edge in sedge_list]
        gens_short = list(set(gens))
        av_lengths = np.zeros(len(gens_short))
        for i in range(len(lengths)):
            av_lengths[gens_short.index(gens[i])] += lengths[i]
        for i in range(len(gens_short)):
            c = len(gens[gens==gens_short[i]])
            av_lengths[i] /= c
        g += gens_short
        l += list(av_lengths)
        t += [i]*len(gens_short)
    plt.scatter(t, g, c=l, cmap='viridis')
    plt.colorbar(label="Largo promedio")
    plt.xlabel("Iteración")
    plt.ylabel("Generación")
    plt.show()

# POCO INTERESANTE
def largo_vs_generation(sample_edges):
    edge_list = sample_edges[2]
    n_nodes = max(edge[1] for edge in edge_list)+1
    generations = np.zeros(n_nodes)
    generations[0] = 0
    generations[1] = 1
    largos = np.zeros(n_nodes)
    for edge in edge_list:
        largos[edge[1]] = edge[2]
    # ASIGNAR GENERACIONES
    while any([g == 0 for g in generations[1:]]):
        for edge in edge_list:
            if generations[edge[0]] > 0:
                generations[edge[1]] = generations[edge[0]]+1
    print(generations)
    print(largos)
    plt.scatter(generations, largos)
    plt.show()

# -------------------

# ---- BULK ----
def bulk_N(Nss, tauss):
    iters = len(Nss[0])
    amnt = len(Nss)
    ts = np.linspace(0, min([taus[-1] for taus in tauss]), iters)
    interpNs = np.empty((amnt, iters))
    for i in range(amnt):
        interpNs[i] = interp1d(tauss[i], Nss[i])(ts)
    av = np.mean(interpNs, 0)
    plt.plot(ts, av)
    # for Ns, taus in zip(Nss, tauss):
    #     plot_N(Ns, taus, show=False)
    plt.show()
# --------------

def get_cambio_fase(Ns):
    w = 40
    idx = -1
    for i in range(w, len(Ns)-w):
        #print(np.argmax(Ns[i-w:i+w]))
        if np.argmax(Ns[i-w:i+w]) == w:
            return i

# data1 = load_data("sims_proba/0.7_0.2_0.1.pickle") # (p0>>p1>p2)
# data2 = load_data("sims_proba/0.7_0.1_0.2.pickle") # (p0>>p2>p1)
# data3 = load_data("sims_proba/0.1_0.7_0.2.pickle") # (p1>>p2>p0)
# data4 = load_data("sims_proba/0.2_0.7_0.1.pickle") # (p1>>p0>p2)
# data5 = load_data("sims_proba/0.1_0.2_0.7.pickle") # (p2>>p1>p0)
# data6 = load_data("sims_proba/0.2_0.1_0.7.pickle") # (p2>>p0>p1)
# data7 = load_data("sims_proba/0.3333333333333333_0.3333333333333333_0.3333333333333333.pickle") # (p2=p0=p1)

# datas = [data1, data2, data3, data4, data5, data6, data7]
# plt.figure(figsize=(10,7), dpi=100)
# labels=["$p_0\\gg p_1>p_2$", "$p_0\\gg p_2>p_1$", "$p_1\gg p_2>p_0$", "$p_1\\gg p_0>p_2$", "$p_2\\gg p_1>p_0$", "$p_2\\gg p_0>p_1$", "$p_0=p_1=p_2$"]
# for i in range(7):
#     t = np.arange(len(Ns))
#     data = datas[i]
#     Ns = data["N"]
#     p0 = data["p0"]
#     p1 = data["p1"]
#     p2 = data["p2"]

#     plt.scatter(t, Ns, marker='.', label=labels[i])
# plt.xlabel("Iteración")
# plt.ylabel("Número de puntas")
# plt.legend()
# plt.savefig("graficos_proba/N(t)_comparado.png")
# plt.show()


# data = load_data("sims_proba/ev_cij2.pickle")

# #p, cov = curve_fit(fit, np.arange(1,len(data["Cij"])), np.log(data["Cij"][1:]), p0=[3.6, -2])
# # print(p)

# scatter_Cijs(data["Cij"], np.arange(len(data["Cij"])), C_=1)


data = load_data("sims/3000_5e-03_6e+04.pickle")
K = data["K"]
l0 = data["l0"]
maxit = data["maxit"]
taus = data["tau"]
Ns = data["N"]
Cijs = data["Cij"]
p0s = data["p0"]
sample_edges = data["sample_edges"]
ls = data["ls"]
taus = data["tau"]
C0 = K/(l0**3)
C_ = data["C_"]
# plt.figure(figsize=(10,7))
# scatter_N(Ns[:1000], taus[:1000])
# plt.figure(figsize=(10,7))
# scatter_l(ls[:1000], taus[:1000], 1,)
plt.figure(figsize=(10,7))
scatter_Cijs(Cijs[:1000], taus[:1000], C_)
# plt.figure(figsize=(10,7))
# plot_Cijs(Cijs, taus, C_, True, "graficos_inst/Cij_vs_tiempo.png")
# plt.figure(figsize=(10,7))
# plot_l(ls, taus, l0, True, "graficos_inst/largo_vs_tiempo.png")
# plt.figure(figsize=(10,7))
# # print(np.argmax(Ns))
# f = get_cambio_fase(Ns)
# print(f, Ns[f])
# plot_N(Ns[:2*f], taus[:2*f])
# plot_probs(p0s[:2*f], taus[:2*f])
# plt.plot(taus[2*f]*np.ones(2), [np.min(Ns[:2*f]), np.max(Ns[:2*f])])
# plt.show()
# # plt.figure(figsize=(10,7))
# plot_probs(p0s, taus, True, "graficos_inst/probs_vs_tiempo.png")
# plot_murray(sample_edges, taus, True, "graficos_inst/murray_lindo.png")
# scatter_v_g(sample_edges[9], v=4)
# terminacion_vs_gen(sample_edges[3])
#print(np.argmax(Ns))
# #scatter_N(Ns[:801], taus[:801])
# plt.figure(figsize=(10,7))
# gens = []
# lengths = []
# print(len(sample_edges))
# for sample in sample_edges[4000:5000:100]:
#     g, l = scatter_v_g(sample, v=4)
#     # plt.show()
#     gens += list(g)
#     lengths += list(l)
# # p, cov = curve_fit(lambda x, a, b: a*x**b, np.array(filterg), filterl, p0=[1, -2])
# # print(p)
# # plt.plot(np.array(gens), (np.array(gens))**(-2)*0.8)
# plt.xlabel("Generación")
# plt.ylabel("Radio promedio")
# # plt.yscale('log')
# # plt.xscale('log')
# plt.savefig("graficos_inst/radio_vs_gen.png")
# plt.show()

def max_N_l_KC():
    plt.figure(figsize=(10,7))

    data = load_data("datos_maxes/q1_l1.pickle")
    Ks = []
    Cs = []
    maxLs = []
    tauLs = []
    maxNs = []
    tauNs = []
    for sim in data:
        Ks.append(sim[0])
        Cs.append(sim[1])
        maxLs.append(sim[2])
        tauLs.append(sim[3])
        maxNs.append(sim[4])
        tauNs.append(sim[5])

    x = np.array(Ks)/np.array(Cs)

    p, cov = curve_fit(fit, np.log(x), np.log(maxNs))

    rth = np.linspace(min(x), max(x), 1000)
    plt.plot(rth, rth**p[0]*np.exp(p[1]), label=f"$y = {np.exp(p[1]):.2f}x^{{{p[0]:.2f}}}$")
    plt.scatter(x, maxNs, label="$q_0=1, l_0=1$")
    data = load_data("datos_maxes/q10_l1.pickle")
    Ks = []
    Cs = []
    maxLs = []
    tauLs = []
    maxNs = []
    tauNs = []
    for sim in data:
        Ks.append(sim[0])
        Cs.append(sim[1])
        maxLs.append(sim[2])
        tauLs.append(sim[3])
        maxNs.append(sim[4])
        tauNs.append(sim[5])
    x = np.array(Ks)/np.array(Cs)
    p, cov = curve_fit(fit, np.log(x), np.log(maxNs))

    rth = np.linspace(min(x), max(x), 1000)
    plt.plot(rth, rth**p[0]*np.exp(p[1]), label=f"$y = {np.exp(p[1]):.2f}x^{{{p[0]:.2f}}}$")
    plt.scatter(x, maxNs, label="$q_0=10, l_0=1$")
    data = load_data("datos_maxes/q1_l5.pickle")
    Ks = []
    Cs = []
    maxLs = []
    tauLs = []
    maxNs = []
    tauNs = []
    for sim in data:
        Ks.append(sim[0])
        Cs.append(sim[1])
        maxLs.append(sim[2])
        tauLs.append(sim[3])
        maxNs.append(sim[4])
        tauNs.append(sim[5])
    x = np.array(Ks)/np.array(Cs)
    p, cov = curve_fit(fit, np.log(x), np.log(maxNs))

    rth = np.linspace(min(x), max(x), 1000)
    plt.plot(rth, rth**p[0]*np.exp(p[1]), label=f"$y = {np.exp(p[1]):.2f}x^{{{p[0]:.2f}}}$")
    plt.scatter(x, maxNs, label="$q_0=1, l_0=5$")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$K/\\bar{C}$")
    plt.ylabel("Máximo de puntas")
    plt.legend()
    plt.savefig("graficos_inst/max_vs_KC.png")
    plt.show()

# max_N_l_KC()