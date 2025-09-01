import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pickle
import copy
import multiprocessing
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import scipy.linalg as la

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

fig, axs = plt.subplots(figsize=(15,15),dpi=100)
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
        (1-qs[n]/maxq, 1-qs[n]/maxq, 1) if n in puntas else 'lightgrey' for n in list(G.nodes())
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

def plot_end(p0s, taus, Cijs, C_, ls, l0, edge_list, puntas, n_nodes, qs, q0, T, K):
    """
    Plotea el sistema final y las evoluciones de <C_ij>, sum(l), N y probabilidades
    """
    fig = plt.figure(figsize=(25, 15), dpi=100)
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1])

    ax_big = fig.add_subplot(gs[:, 0])

    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])
    # PROBS
    p0smooth = exponential_moving_average(p0s, 0.05)

    ax1.scatter(taus, p0s, marker='.', color='lightblue', alpha=0.3)
    ax1.scatter(taus, 1 - p0s, marker=".", color='peachpuff', alpha=0.3)
    ax1.plot(taus, p0smooth, label="Retracción")
    ax1.plot(taus, 1 - p0smooth, label="Crecimiento + Bifurcación")
    ax1.set_xlabel("t")
    ax1.set_ylabel("Probabilidad")
    ax1.legend(loc='upper right')

    # CIJS
    Csmooth = exponential_moving_average(Cijs, 0.05)
    ax2.set_yscale('log')
    ax2.scatter(taus, Cijs, marker='.', color='lightblue', alpha=0.3)
    ax2.plot(taus, Csmooth, label="$<C_{ij}>$ de las puntas")
    ax2.plot(taus, C_*np.ones(len(Cijs)), label="$\\bar{C}$")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Conductancia")
    ax2.legend()

    # LS
    smoothls = exponential_moving_average(ls, 0.05)
    ax3.scatter(taus, ls/l0, marker='.', color='lightblue', alpha=0.3)
    ax3.plot(taus, smoothls/l0)
    ax3.set_xlabel("t")
    ax3.set_ylabel("Largo total")

    # GRAPH FINAL
    show_graph(edge_list, puntas, n_nodes, qs, q0, 0, T, K, C_, ax_big)

def Cij(r, mu, l):
    """
    Calcula el Cij de un ducto
    """
    return np.pi*r**4 / (8*mu*l)

def get_path(n, padres, p):
    """
    Encuentra el padre de un nodo
    """
    padre = padres[n]
    if padre == 0:
        return p
    else:
        p.append(padre)
        return get_path(padre, padres, p)
    
def optimize_Cij(edge_list, K, puntas):
    """
    Optimiza los Cij, modifica directamente edge_list y retorna el promedio de las puntas
    """
    N = 1
    if len(edge_list) == 1:
        edge_list[0][4] = K / (edge_list[0][2]**3)
        return edge_list[0][4]
    den_sum = sum([np.mean(edge[3])**(1/3)*edge[2] for edge in edge_list])
    av = 0

    for edge in edge_list:
        edge[4] = np.mean(edge[3])**(2/3)*K / (den_sum**2 * edge[2]) # OPTIMIZACION INSTANTANEA
        #edge[4] += (np.mean(edge[3])**(2/3)*K / (den_sum**2 * edge[2]) - edge[4])/N
        if edge[1] in puntas:
            av += edge[4]
    return av/len(puntas)

def get_in_edge(n, edge_list):
    """
    Encuentra el ducto entrante a un nodo
    """
    for edge in edge_list:
        if edge[1] == n:
            return edge

def probs(Cijs, C_):
    """
    Calcula p1, p0 y p2 se deducen
    """
    p1 = (1 - np.exp(-Cijs/C_))/2
    return p1

def Laplacian(edge_list, n_nodes):
        """
        Calcula la matriz Laplaciana de conductancias
        """
        L = np.zeros((n_nodes, n_nodes))
        for edge in edge_list:
            c = edge[4]
            L[edge[0], edge[1]] = -c
            L[edge[1], edge[0]] = -c
            L[edge[0], edge[0]] += c
            L[edge[1], edge[1]] += c
        return L

def calc_volume(edge_list):
    """
    Calcula el volumen total del sistema
    """
    V = 0
    for edge in edge_list:
        V += (edge[4] * edge[2])**(1/2) * edge[2]
    return V

def positive_noise(N, mu, sigma=1):
    """
    Genera N ruidos blancos positivos
    """
    result = []
    batch_size = int(N * 1.5)

    while len(result) < N:
        samples = np.random.normal(mu, sigma, batch_size)
        positives = samples[samples >= 0]
        result.extend(positives.tolist())
    if N == 1:
        return result[0]
    return np.array(result[:N])


def sim(K=15000,
        q0=1,
        l0=1,
        maxit=50000,
        animate=False,
        C_=0.2,
        frames=10,
        save_graphs=False):
    if K==0:return []
    np.random.seed(os.getpid())
    mu = 0.001 # agua
    r0 = 100
    C0 = K/(l0**3)
    t0 = 1 # Tiempo que toma en crecer de l0
    v0 = l0/t0
    P = 10 # sobre cuántas iteraciones se promedia el flujo
    # Condiciones iniciales
    n_nodes = 2
    # n1 -> n2, lij, [Qij**2]*5, Cij
    edge_list = [[0, 1, l0*(positive_noise(1,1,1/3)), [q0**2]*P, C0]] # Inicializado a 2 nodos conectados
    puntas = [1] # Puntas del arbol (con inflow)
    padres = [-1, 0] # Guardamos el nodo padre de cada nodo

    Ns = np.zeros(maxit) # num de puntas
    Ns[0] = 1
    Cijs = np.zeros(maxit)
    Cijs[0] = C0
    p0s = np.zeros(maxit)
    p1s = np.zeros(maxit)
    p2s = np.zeros(maxit)
    p1s[0] = probs(edge_list[0][4], C_)
    p2s[0] = p1s[0]
    p0s[0] = 1-2*p1s[0]
    taus = np.zeros(maxit)
    #taus = np.linspace(0, 8000, 80000)
    ls = np.zeros(maxit)
    ls[0] = edge_list[0][2]
    T = 0
    sample_edges = []

    # -------------------------
    # SIMULACION
    # -------------------------
    start = time.time()
    for j in range(1,maxit):
        tau = -np.log(np.random.rand())/len(puntas)
        T += tau
        taus[j] = T
    # ----------------
    # EVOLUCION
    # ----------------
        # Elegimos el nodo a modificar
        n = np.random.choice(puntas)
        # Recuperamos el ducto entrante (ducto padre?)
        edge_n = get_in_edge(n, edge_list)
        # Elegimos la acción
        prob = np.random.rand()

        # PROBABILIDADES
        Cijs_puntas = np.zeros(len(puntas))
        for edge in edge_list:
            if edge[1] in puntas:
                Cijs_puntas[puntas.index(edge[1])] = edge[4]
        # ORDENADAS COMO puntas
        p1l = probs(Cijs_puntas, C_)
        p2l = p1l.copy()
        p0l = 1 - 2*p1l
        p0s[j] = np.mean(p0l)
        p1s[j] = np.mean(p1l)
        p2s[j] = np.mean(p2l)

        ps = np.concatenate((p0l, p1l, p2l)) / len(puntas)
        suma = np.cumsum(ps)
        r = np.random.rand()
        for i in range(len(suma)):
            if suma[i] >= r:
                n = puntas[i%len(puntas)]
                accion = i//len(puntas)
                break
        edge_n = get_in_edge(n, edge_list)

        # RETRACCION
        #if prob <= p0:
        if accion==0:
            # Acortamos el ducto
            old = edge_n[2]
            #edge_n[4] += tau*edge_n[4]*5*v0/edge_n[2] # PAULATINO
            #edge_n[2] -= v0*tau*positive_noise(1,1,1/3)
            edge_n[2] -= l0*positive_noise(1,1,1/3)
            edge_n[4] = edge_n[4] * old / edge_n[2] # escalado
            if edge_n[2] <= l0/2:
                if n_nodes == 2:
                    print("MURIO")
                    break

                # MUERE
                puntas.remove(n)
                edge_list.remove(edge_n)
                padre = edge_n[0]

                # BUSCA CONEXIONES PADRE
                entrada, salida = None, None
                for edge in edge_list:
                    if edge[1] == padre:
                        entrada = edge
                    if edge[0] == padre and edge[1] != n:
                        salida = edge
                    if entrada and salida: break
                # FUSION DE EDGES -> Los Cij se suman como resistencias en serie?
                edge_list.append( [entrada[0], salida[1], entrada[2]+salida[2], [q0**2]*P, (salida[4]+entrada[4])/2] )

                padres[salida[1]] = entrada[0]
                edge_list.remove(entrada)

                edge_list.remove(salida)

                padres.pop(n)
                padres.pop(padre)
                # Hacemos un shift de los nodos con indice mayor a n o padre: Si mayor a padre -1, si mayor a n -1 tmb
                for edge in edge_list:
                    if edge[0] >= padre:
                        edge[0] -= 1
                        if edge[0] >= n:
                            edge[0] -= 1
                    if edge[1] >= padre:
                        edge[1] -= 1
                        if edge[1] >= n:
                            edge[1] -= 1
                for i in range(len(puntas)):
                    if puntas[i] >= padre:
                        puntas[i] -= 1
                        if puntas[i] >= n:
                            puntas[i] -= 1
                for i in range(len(padres)):
                    if padres[i] >= padre:
                        padres[i] -= 1
                        if padres[i] >= n:
                            padres[i] -= 1
                n_nodes -= 2
        
        # CRECIMIENTO
        #elif prob <= p1+p0:
        elif accion==1:
            old = edge_n[2]
            #edge_n[4] -= tau*edge_n[4]*5*v0/edge_n[2]
            #edge_n[2] += v0*tau*positive_noise(1,1,1/3)
            edge_n[2] += l0*positive_noise(1,1,1/3)
            edge_n[4] = edge_n[4] * old / edge_n[2]
        # BIFURCACION
        else:
            # Los r de los hijos = r padre -> Escalamos el Cij al largo
            #l1 = v0*tau*positive_noise(1,1,1/3)
            l1 = l0*positive_noise(1,1,1/3)

            edge_list.insert(0,[n, n_nodes, l1, edge_n[3].copy(), edge_n[4]*edge_n[2]/l1])
            puntas.append(n_nodes)
            padres.append(n)
            #l2 = v0*tau*positive_noise(1,1,1/3)
            l2 = l0*positive_noise(1,1,1/3)
            edge_list.insert(0,[n, n_nodes+1, l2, edge_n[3].copy(), edge_n[4]*edge_n[2]/l2])
            puntas.append(n_nodes+1)
            padres.append(n)

            puntas.remove(n)
            n_nodes += 2



    # --------------------
    # OPTIMIZACION
    # --------------------
        # Generar inflows en puntas
        qs = np.zeros(n_nodes)
        #qs[puntas] = q0*positive_noise(len(puntas),1,1/3)
        qs[puntas] = np.random.choice([0, 2*q0], len(puntas))
        qs[0] = -np.sum(qs)
        """ FORMA SIMPLIFICADA POR TOPOLOGIA DE ARBOL
        # Por la topología del arbol (sin loops), podemos calcular los Qij (flujo interno) como sumas de los inflows por todo el arbol.
    
        # Calculamos el camino punta-sumidero para cada punta:
        paths = []
        for punta in puntas:
            paths.append(get_path(punta, padres, [punta]))
        # Calculamos el flujo q_i (interno) en cada nodo
        for path in paths:
            if len(path) == 1:
                continue
            q_punta = qs[path[0]]
            for node in path[1:]:
                qs[node] += q_punta

        l = 0
        # Nuevamente, por la topología: Q_ij = q_j (viendo j como el nodo hijo) PROMEDIAMOS SOBRE 5 ITERACIONES
        for edge in edge_list:
            edge[3].append(qs[edge[1]]**2)
            edge[3].pop(0)
            l += edge[2]
        ls[j] = l
        """

        
        """ FORMA GENERAL (MAS LENTA)"""
        # Para mantenerlo general, lo hacemos con la matriz Laplaciana L:
        L = Laplacian(edge_list, n_nodes)
        invL = la.pinv(L, atol=1e-8)
        ps = np.matmul(invL, qs)
        # Calcular Qij
        l = 0
        for edge in edge_list:
            Qij = edge[4] * (ps[edge[1]] - ps[edge[0]])
            edge[3].append(Qij**2)
            edge[3].pop(0)
            l += edge[2]
        ls[j] = l
        #"""
        
        # Optimizamos los Cij según la formula, guardamos el promedio para análisis
        Cijs[j] = optimize_Cij(edge_list, K, puntas)

        if animate:# and j%1000==0:# and j%(maxit/frames) == 0:
            print("ANIMATING: ", j)
            show_graph(edge_list, puntas, n_nodes, qs, q0, j, T, K, C_, axs, True)

        if j%1000 == 0:
            if save_graphs:
                show_graph(edge_list, puntas, n_nodes, qs, q0, j, T, K, C_, axs, save=True, path=f"arboles/{K}_{C_}_{j}.png")
            print(j, n_nodes, T, Cijs[j], np.min(Cijs_puntas), ls[j])

        # GUARDAR INFO
        Ns[j] = len(puntas)


        if j % (maxit / 5000)==0:
            sample_edges.append(copy.deepcopy(edge_list))
        # if j%100==0:

    end = time.time()
    print("TOTAL:", end - start)
    #plot_end(p0s, taus, Cijs, C_, ls, l0, edge_list, puntas, n_nodes, qs, q0, T, K)
    data = {
            "K": K,
            "l0": l0,
            "maxit": maxit,
            "tau": taus,
            "N": Ns, # N PUNTAS
            "Cij": Cijs,
            "p0": p0s,
            "ls": ls,
            "sample_edges": sample_edges,
            "C_": C_
        }
   
    return data

def save_sim():
    K = 10
    maxit = 60000
    C_ = 1e-3
    data = sim(C_=C_, K=K, maxit = maxit)
    base_name = f"./sims/{K}_{C_:.0e}_{maxit:.0e}"
    name = base_name + ".pickle"
    counter = 1
    while os.path.exists(name):
        name = base_name + f"_{counter}"+".pickle"
        counter +=1
    print(f"Saving as: {name}")
    with open(name, "wb") as f:
        pickle.dump(data, f)
save_sim()
def parallel_sim(q, kwargs):
    data = sim(**kwargs)
    q.put(data)

def parallel_save(q, K, C_, maxit, i):
    data = sim(K=K, C_=C_, maxit=maxit)
    with open(f"./K_1000_5e-3/{i}.pickle", "wb") as f:
        pickle.dump(data, f)
    q.put("DONE")

def parallel_bulk():
    # MODIFICAR SEGUN NECESARIO
    q = multiprocessing.Manager().Queue()
    processes = []
    N = 20
    K = 1000
    C_ = 5e-3
    datas = []
    active = []
    for i in range(N):
        print(i)
        p = multiprocessing.Process(target=parallel_save, args=(q, K, C_, 30000, i))
        p.start()
        active.append(p)
        if len(active) == 30:
            active[0].join()
            active.pop(0)
    for p in active:
        p.join()        

    print("ALL STARTED")
    print("ALL DONE")
    for _ in range(N):
        status = q.get()
        print(status)

def gen_Nmax_vs_C_():
    q = multiprocessing.Manager().Queue()
    processes = []
    N = 200
    maxit = 1000
    sims = []
    active = []
    i=0
    for K in Ks:
        for C in C_s:
            print(i)
            i+=1
            p = multiprocessing.Process(target=parallel_sim, args=(q, {"K":K, "maxit":1000, "C_":C}))
            p.start()
            active.append(p)
            if len(active) == 30:
                active[0].join()
                active.pop(0)
    for p in active:
        p.join()

    for K in Ks:
        for C in C_s:
            result = q.get()
            maxLi = np.argmax(result["ls"])
            maxL = result["ls"][maxLi]
            tauL = result["tau"][maxLi]
            maxNi = np.argmax(result["N"])
            maxN = result["N"][maxNi]
            tauN = result["tau"][maxNi]
            print(f"DONE {K}, {C}")
            sims.append((K, C, maxL, tauL, maxN, tauN))
    print("DONE")
    with open("./datos_maxes.pickle", "wb") as f:
        pickle.dump(sims, f)

def gen_K_C():
    Ks = [1000]
    fracs = np.logspace(3, 6, 60, base=10)
    sims = []
    for K in Ks:
        for f in fracs:
            C_ = K/f
            maxL = 0
            tauL = 0
            maxN = 0
            tauN = 0
            for _ in range(3):
                maxit = 1000
                data = sim(C_=C_, K=K, maxit = maxit, q0=1, l0=2)
                maxLi = np.argmax(data["ls"])
                maxL += data["ls"][maxLi]
                tauL += data["tau"][maxLi]
                maxNi = np.argmax(data["N"])
                maxN += data["N"][maxNi]
                tauN += data["tau"][maxNi]
            maxL /= 3
            maxN /= 3
            tauL /= 3
            tauN /= 3
            print(f"DONE {K}, {C_}")
            sims.append((K, C_, maxL, tauL, maxN, tauN))
    with open("./datos_maxes/q1_l5.pickle", "wb") as f:
        pickle.dump(sims, f)

#gen_K_C()

print("\a")