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

def run_simulation(gen, nights_per_gen, repeats_per_night, payoff_saturation_timepoint,
                   rootN, filledprop, cost, r, K, k, rang, start_cfreq):
    
    nm = int((rootN ** 2) * filledprop)
    nf = int(nm / 2)
    matelist_3d = np.zeros((gen, nights_per_gen, nf))
    
    for i in range(gen):
        if i == 0:
            coopprop = start_cfreq
        else:
            matelist_2d = matelist_3d[i - 1, :, :]
            num_coop = np.count_nonzero(matelist_2d == -1)
            coopprop = num_coop / matelist_2d.size
        
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
                #mov_phase, mov_strat = mov(next_phase, init_strat, init_pay, next_pay, k)
                mov_phase, mov_strat = mov2(next_phase, init_strat, next_pay, cost)
                init_phase = mov_phase
                init_strat = mov_strat
                init_pay = payoff_arr(init_phase, init_strat, rang, cost)

            for t in range(payoff_saturation_timepoint, repeats_per_night):
                if t == next_fem_time:
                    mate_strat = mate_choice(init_pay, init_strat)
                    matelist_3d[i, j, mate_count] = mate_strat
                    mate_count += 1
                    fem_idx += 1
                    if fem_idx < nf:
                        next_fem_time = fem_timepoints[fem_idx]

    # Extract cooperator proportions
    coopprop_values = []
    for i in range(gen):
        matelist_2d = matelist_3d[i, :, :]
        num_coop = np.count_nonzero(matelist_2d == -1)
        tot = matelist_2d.size
        coopprop = num_coop / tot
        coopprop_values.append(coopprop)
    
    coopprop_values.insert(0, start_cfreq)
    return np.array(coopprop_values)


def run_survival_trial(mutant_type, start_freq, trial_id,
                     gen,
                     nights_per_gen,
                     repeats_per_night,
                     payoff_saturation_timepoint,
                     rootN,
                     filledprop,
                     cost,
                     r,
                     K,
                     k,
                     rang):

    #seed = hash((mutant_type, start_freq, trial_id)) % 2**32
    #np.random.seed(seed)
    #random.seed(seed)
    base_seed = 12345

    seed = (
        base_seed
        + trial_id
        + int(start_freq * 1e6)
        + (0 if mutant_type == "coop" else 10_000_000)
    )

    np.random.seed(seed)
    random.seed(seed)

    coopprops = run_simulation(gen=gen, nights_per_gen=nights_per_gen,
                     repeats_per_night=repeats_per_night,
                     payoff_saturation_timepoint=payoff_saturation_timepoint,
                     rootN=rootN,
                     filledprop=filledprop,
                     cost = cost,
                     r=r,
                     K=K,
                     k=k,
                     rang=rang,
                     start_cfreq=start_freq)
        
    final_p = coopprops[-1]
    if mutant_type == "coop":
        survived = final_p >= 0.005
    else:
        survived = final_p <= 0.995

    return mutant_type, start_freq, survived

def survival_main():

    start_freqs_coop = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    start_freqs_def = [1 - f for f in start_freqs_coop]

    n_trials = 30

    tasks = []

    for f in start_freqs_coop:
        for t in range(n_trials):
            tasks.append(("coop", f, t))

    for f in start_freqs_def:
        for t in range(n_trials):
            tasks.append(("def", f, t))

    results = {"coop": {f: [] for f in start_freqs_coop}, "def": {f: [] for f in start_freqs_def}}
    n_workers = max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn")
    ) as executor:

        futures = [
            executor.submit(run_survival_trial, mutant_type, f, trial_id, 10, 30, 10000, 75, 20, 0.5, 0.05, 1, 0.5, 1000, 1)
            for mutant_type, f, trial_id in tasks
        ]

        for fut in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Running survival trials"):
            mutant_type, f, survived = fut.result()
            results[mutant_type][f].append(survived)

    return results, start_freqs_coop, start_freqs_def  # <--- return results for plotting

if __name__ == "__main__":
    results, start_freqs_coop, start_freqs_def = survival_main()

    coop_survival_probs = [np.mean(results["coop"][f]) for f in start_freqs_coop]
    def_survival_probs = [np.mean(results["def"][f]) for f in start_freqs_def]
    
    import matplotlib.pyplot as plt
    x_coop = start_freqs_coop
    x_def = [1 - f for f in start_freqs_def]

    plt.figure(figsize = (8, 5))
    plt.plot(x_coop, coop_survival_probs, marker = 'o', linestyle = '-', color = 'red', label = 'cooperator', linewidth = 3)
    plt.plot(x_def, def_survival_probs, marker = 'o', linestyle = '-', color = 'blue', label = 'defector', linewidth = 2)

    plt.xlabel('initial mutant frequency', fontsize = 22)
    plt.ylabel('probability of survival', fontsize = 22)
    #plt.ylim(0,1)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)  
    plt.legend(fontsize = 20)
    plt.grid(False)

    plt.tight_layout()
    plt.show()

