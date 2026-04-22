import numpy as np
import seaborn as sns
import random
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import concurrent.futures
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


from matplotlib.colors import ListedColormap
from IPython.display import HTML
from collections import defaultdict

from matplotlib.patches import Patch
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

#functions

#function generates lattice of 1s and 0s; 1 means there's a firefly there, 0 means it's empty
def lattice(rows, cols, prop_one): #prop_one is proportion of ones
    tot_cells = rows * cols
    num_ones = int(tot_cells * prop_one)

    arr = np.zeros((rows, cols))

    ind_ones = np.random.choice(tot_cells, num_ones, replace = False)
    row_ind = ind_ones // cols
    col_ind = ind_ones % cols

    arr[row_ind, col_ind] = 1

    return arr

#function generates list of firefly phases between 0 and 2pi 
def phase_list(num_ff):
    phases = []
    while len(phases) < num_ff:
        phase = random.uniform(0, 2 * math.pi)
        if phase != 0 and phase != 2 * math.pi:
            phases.append(phase)
    return phases

#function assigns non-empty lattice positions to firefly phases from list
def phase_arr(positions, phases):
    phase_array = np.zeros_like(positions)
    occ_indices = np.argwhere(positions) #list of indices with non-zero entries

    count = 0

    for i, j in occ_indices:
        phase_array[i, j] = phases[count]
        count = count + 1
    return phase_array          

#strat assigning function
def altstrat(positions, propC):
    num_nz = np.count_nonzero(positions)
    num_coop = round(num_nz * propC)
    num_def = num_nz - num_coop

    altstrat = np.zeros_like(positions)
    nz_indices = np.argwhere(positions)
    ord_pairs = [tuple(index) for index in nz_indices]

    indices_coop = np.random.choice(len(ord_pairs), size = num_coop, replace = False)
    loc_coop = [ord_pairs[index_coop] for index_coop in indices_coop]

    rem_pairs = [pair for pair in ord_pairs if pair not in loc_coop]
    indices_def = np.random.choice(len(rem_pairs), size = num_def, replace = False)
    loc_def = [rem_pairs[index_def] for index_def in indices_def]

    for i, j in loc_coop:
        altstrat[i, j] = -1
    
    for i, j in loc_def:
        altstrat[i, j] = -2
    
    return altstrat

#function gives phase-order (absolute value of sum of non-zero phasors/number of non-zero phasors) of phase list
def cohs(numlist):
    compexp_list = [cmath.exp(1j * x) for x in numlist]
    total = abs(sum(compexp_list))
    phaseorder = total/9
    return phaseorder

#function gives list of phases of focal cell and its non-zero neighbors
def neighbors(a, radius, r, c):
    numlist = [ a[i, j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
                for j in range(c - 1 - radius, c + radius)
                    for i in range(r - 1 - radius, r + radius) ]
    non_zero_nghbrs = [x for x in numlist if x != 0]
    return non_zero_nghbrs

#function generates payoff array, ones in the bulk can have a maximum of 8 neighbors so maxbright = 1.
#ones on the sides can have max 5 neighbors so maxbright = 6/9 = 2/3
#ones on the corners can have max 3 neighbors so maxbright = 4/9
def payoff_arr(ph_arr, strat_arr, rang, cost):
    pay = np.zeros_like(ph_arr)
    filled_indices = np.argwhere(ph_arr)

    for i, j in filled_indices:
        k = neighbors(ph_arr, rang, i + 1, j + 1)
        if strat_arr[i, j] == -1:
            pay[i, j] = cohs(k)
        else:
            pay[i, j] = cohs(k) + cost
    return pay
       
#kuramoto
def kur3(phase, strat, r, K):
    phasenext = np.zeros_like(phase)
    fireflylocs = [(i, j) for i, j in np.argwhere(phase)]
    random.shuffle(fireflylocs)
    
    for i, j in fireflylocs:
        if strat[i, j] == -1:
            nehbrs = neighbors(phase, r, i + 1, j + 1)
            templist = [np.sin(nehbr - phase[i, j]) for nehbr in nehbrs]
            if templist:
                cont = sum(templist)/len(templist)
            else:
                cont = 0.0
            phasenext[i, j] = (phase[i, j] + (K * cont)) % (2 * math.pi)
        else:
            phasenext[i, j] = phase[i, j]
    return phasenext

#function updates position, and carries phase and strat to new position too
def mov(phase, strat, old_pay, new_pay, k):
    pay_diff = np.subtract(new_pay, old_pay)
    filled_indices = [(i, j) for i, j in np.argwhere(strat)]
    random.shuffle(filled_indices)

    for i, j in filled_indices:
        random_num = np.random.rand()
        prob = (1 + math.tanh(k * pay_diff[i, j]))/2 #probability of staying in same location
        if prob < random_num:
            neigbors = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
            neigbors_inbound = [(x, y) for x, y in neigbors if 0 <= x < strat.shape[0] and 0 <= y < strat.shape[1]]
            neigbors_empty = [(x, y) for x, y in neigbors_inbound if strat[x, y] == 0]
            if neigbors_empty:
                new_i, new_j = neigbors_empty[np.random.randint(len(neigbors_empty))]
                phase[new_i, new_j] = phase[i, j] #print phases before and after
                strat[new_i, new_j] = strat[i, j]
                phase[i, j] = 0
                strat[i, j] = 0
    return phase, strat 

#alternate movement function
def mov2(phase, strat, pay, cost):
    filled_indices = [(i, j) for i, j in np.argwhere(strat)]
    random.shuffle(filled_indices)

    for i, j in filled_indices:
        neig = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
        neig_inbound = [(x, y) for x, y in neig if 0 <= x < strat.shape[0] and 0 <= y < strat.shape[1]]
        neig_empty = [(x, y) for x, y in neig_inbound if strat[x, y] == 0]
        if neig_empty:
            if strat[i, j] == -1:
                payloc = [(cohs(neighbors(phase, 1, x + 1, y + 1) + [phase[i, j]]), (x, y)) for x, y in neig_empty]
                max_payloc = max(payloc, key=lambda item: item[0])
                if max_payloc[0] > pay[i, j]:
                    new_i, new_j = max_payloc[1]
                    phase[new_i, new_j] = phase[i, j]
                    strat[new_i, new_j] = strat[i, j]
                    phase[i, j] = 0
                    strat[i, j] = 0
            if strat[i, j] == -2:
                payloc = [(cohs(neighbors(phase, 1, x + 1, y + 1) + [phase[i, j]]) + cost, (x, y)) for x, y in neig_empty]
                max_payloc = max(payloc, key=lambda item: item[0])
                if max_payloc[0] > pay[i, j]:
                    new_i, new_j = max_payloc[1]
                    phase[new_i, new_j] = phase[i, j]
                    strat[new_i, new_j] = strat[i, j]
                    phase[i, j] = 0
                    strat[i, j] = 0
    return phase, strat

#function that chooses (based on payoff values) a single male for mating
def mate_choice(payoff_array, strat_array):
    non_zero_sum = np.sum(payoff_array[payoff_array != 0]) #sum of payoffs
    mateprob_array = payoff_array / non_zero_sum #normalized fitness payoff, equal to mating probability

    non_zero_indices = np.transpose(np.nonzero(mateprob_array)) #indices of fireflies
    non_zero_values = mateprob_array[mateprob_array.nonzero()] #probabilities

    chosen_index = np.random.choice(len(non_zero_values), p = non_zero_values) #making random weighted choice
    chosen_indices = non_zero_indices[chosen_index] #index of chosen male

    chosen_strat = strat_array[chosen_indices[0], chosen_indices[1]] #strategy of chosen male

    return chosen_strat


def run_single_trial(cost, trial_id,
                     gen=50,
                     nights_per_gen=30,
                     repeats_per_night=10000,
                     payoff_saturation_timepoint=75,
                     rootN=20,
                     filledprop=0.5,
                     r=1,
                     K=0.5,
                     k=1000,
                     rang=1,
                     start_cfreq=0.5):

    seed = (os.getpid() * 100000) + trial_id
    np.random.seed(seed)
    random.seed(seed)

    nm = int((rootN ** 2) * filledprop)
    nf = int(nm / 2)

    coopprop_values = [start_cfreq]
    matelist_3d = np.zeros((gen, nights_per_gen, nf), dtype=np.int8)

    for i in range(gen):
        # Update coopprop from previous generation
        if i == 0:
            coopprop = start_cfreq
        else:
            matelist_2d = matelist_3d[i - 1]
            coopprop = np.count_nonzero(matelist_2d == -1) / matelist_2d.size

        for j in range(nights_per_gen):
            init_pos = lattice(rootN, rootN, filledprop)
            init_phase_list = phase_list(nm)
            init_phase = phase_arr(init_pos, init_phase_list)
            init_strat = altstrat(init_pos, coopprop)
            init_pay = payoff_arr(init_phase, init_strat, rang, cost)

            fem_timepoints = np.random.choice(repeats_per_night, nf, replace=False)
            fem_timepoints.sort()
            mate_count = 0
            fem_idx = 0
            next_fem_time = fem_timepoints[fem_idx]

            for t in range(payoff_saturation_timepoint):
                if t == next_fem_time:
                    mate_strat = mate_choice(init_pay, init_strat)
                    matelist_3d[i, j, mate_count] = mate_strat
                    mate_count += 1
                    fem_idx += 1
                    if fem_idx < nf:
                        next_fem_time = fem_timepoints[fem_idx]

                next_phase = kur3(init_phase, init_strat, r, K)
                next_pay = payoff_arr(next_phase, init_strat, rang, cost)
                mov_phase, mov_strat = mov(next_phase, init_strat, init_pay, next_pay, k)
                #mov_phase, mov_strat = mov2(next_phase, init_strat, next_pay, cost)
                init_phase, init_strat = mov_phase, mov_strat
                init_pay = payoff_arr(init_phase, init_strat, rang, cost)

            for t in range(payoff_saturation_timepoint, repeats_per_night):
                if t == next_fem_time:
                    mate_strat = mate_choice(init_pay, init_strat)
                    matelist_3d[i, j, mate_count] = mate_strat
                    mate_count += 1
                    fem_idx += 1
                    if fem_idx < nf:
                        next_fem_time = fem_timepoints[fem_idx]

    # Final coopprop time series
    for i in range(gen):
        matelist_2d = matelist_3d[i]
        coopprop = np.count_nonzero(matelist_2d == -1) / matelist_2d.size
        coopprop_values.append(coopprop)

    return cost, trial_id, np.array(coopprop_values)

def main():
    cost_values = np.round(np.arange(0.01, 0.12, 0.01), 2)
    n_trials = 10

    tasks = [(c, t) for c in cost_values for t in range(n_trials)]
    results = {c: [] for c in cost_values}

    n_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn")
    ) as executor:

        futures = [
            executor.submit(run_single_trial, cost, trial)
            for cost, trial in tasks
        ]

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Running simulations"):
            cost, trial_id, coop_series = future.result()
            results[cost].append(coop_series)

    return results, cost_values  # <--- return results for plotting

if __name__ == "__main__":
    results, cost_values = main()

    gen = len(next(iter(results.values()))[0]) - 1  # get number of generations from coopprop length

    mean_results = {}
    sem_results = {}

    for cost, trials in results.items():
        data = np.vstack(trials)  # shape: (n_trials, gen+1)
        mean_results[cost] = data.mean(axis=0)
        sem_results[cost] = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    colors = ["red", "purple", "blue"]
    cmap = LinearSegmentedColormap.from_list("RedPurpleBlue", colors)

    norm = mcolors.Normalize(vmin = min(cost_values), vmax = max(cost_values))

    x = np.arange(gen + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for cost in cost_values:
        mean = mean_results[cost]
        sem = sem_results[cost]

        color = cmap(norm(cost))

        ax.plot(x, mean, color=color, label=f'cost = {cost}')
        ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.25
        )

    ax.set_xlabel('generation (T)', fontsize=22)
    ax.set_ylabel('cooperator frequency (p)', fontsize=22)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize = 20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax = ax)
    cbar.set_label('cost', fontsize=22)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label_position('left')

    plt.tight_layout()
    plt.show()

