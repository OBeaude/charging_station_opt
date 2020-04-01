# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:38:25 2020

@author: B57876
"""

import numpy as np
# perso
from optim_cs_v2g_global_v2 import OptimV2gGlobal, calc_load_cost
from optim_cs_v2g import OptimV2g
from optim_cs_v1g import OptimV1g

solver = "COIN"
time_limit_s = 60

def compare_charging_strat(T, n_ev, t_dep, t_arr, p_max_evs, init_soc, dep_soc,
                          p_max_cs, step_time, elec_price):
    
    """
    Compare different charging strategies on a daily charging pb at the station
    :param T: nber of time-slots
    :param n_ev: nber of EVs
    :param t_dep: vector of departure time-slots (of size n_ev)
    :param t_arr: idem for arrival time-slots (of size n_ev)
    :param p_max_evs: per EV maximal charging power (of size n_ev)
    :param init_soc: initial SOC à t=0 (of size n_ev)
    :param dep_soc: needed SOC at departure time (of size n_ev)
    :param p_max_cs: maximal power allowed at the charging station
    :param step_time: time step duration, in s
    :param elec_price: vector of electricity prices (of size T)
    """
    
    # V1G
    print("V1G")
    print("before departure")
    pb_v1g_dep = OptimV1g(True, T, t_dep, np.zeros(n_ev), p_max_evs, init_soc, dep_soc,
                          p_max_cs, step_time, elec_price, solver, time_limit_s)
    ev_load_opt_v1g_dep = pb_v1g_dep.solve()

    cost_opt_v1g_dep = calc_load_cost(ev_load_opt_v1g_dep, 
                                      elec_price[:ev_load_opt_v1g_dep.shape[1]])

    # second phase - after arrival
    print("after arrival")
    pb_v1g_arr = OptimV1g(False, T, T*np.ones(n_ev), t_arr, p_max_evs, 
                          init_soc + step_time/3600*np.sum(ev_load_opt_v1g_dep, axis=1) \
                          - e_drive, np.zeros(n_ev), p_max_cs, step_time,
                          elec_price, solver, time_limit_s)
    ev_load_opt_v1g_arr = pb_v1g_arr.solve()

    cost_opt_v1g_arr = calc_load_cost(ev_load_opt_v1g_arr, 
                                      elec_price[-ev_load_opt_v1g_arr.shape[1]:])
    print("##\nCost V1G = %.2f\n##" % (cost_opt_v1g_dep+cost_opt_v1g_arr))
    
    # concatenate results
    ev_load_opt_v1g = np.zeros([n_ev,T])
    ev_load_opt_v1g[:,:ev_load_opt_v1g_dep.shape[1]] = ev_load_opt_v1g_dep
    ev_load_opt_v1g[:,-ev_load_opt_v1g_arr.shape[1]:] = ev_load_opt_v1g_arr
    
    # V2G
    print("V2G")
    # first phase - before departure
    print("before departure")
    pb_v2g_dep = OptimV2g(True, T, t_dep, np.zeros(n_ev), p_max_evs, 
                          init_soc, dep_soc, soc_max, p_max_cs, step_time, 
                          elec_price, solver, time_limit_s)
    ev_load_opt_v2g_dep = pb_v2g_dep.solve()
    
    cost_opt_v2g_dep = calc_load_cost(ev_load_opt_v2g_dep, 
                                      elec_price[:ev_load_opt_v2g_dep.shape[1]])
    
    # second phase - after arrival
    print("after arrival")
    pb_v2g_arr = OptimV2g(False, T, T*np.ones(n_ev), t_arr, p_max_evs, 
                          init_soc + step_time/3600*np.sum(ev_load_opt_v2g_dep, axis=1) \
                          - e_drive, np.zeros(n_ev), soc_max, p_max_cs, 
                          step_time, elec_price, solver, time_limit_s)
    ev_load_opt_v2g_arr = pb_v2g_arr.solve()

    cost_opt_v2g_arr = calc_load_cost(ev_load_opt_v2g_arr, 
                                      elec_price[-ev_load_opt_v2g_arr.shape[1]:])
    print("##\nCost V2G = %.2f\n##" % (cost_opt_v2g_dep+cost_opt_v2g_arr))

    # concatenate results
    ev_load_opt_v2g = np.zeros([n_ev,T])
    ev_load_opt_v2g[:,:ev_load_opt_v2g_dep.shape[1]] = ev_load_opt_v2g_dep
    ev_load_opt_v2g[:,-ev_load_opt_v2g_arr.shape[1]:] = ev_load_opt_v2g_arr

    # V2G global
    print("V2G global")
    pb_v2g_glob = OptimV2gGlobal(T, t_dep, t_arr, p_max_evs, init_soc, dep_soc,
                                e_drive, soc_max, p_max_cs, step_time, elec_price, 
                                solver, time_limit_s)
    ev_load_opt_v2g_glob = pb_v2g_glob.solve()
    
    cost_opt_v2g_glob = calc_load_cost(ev_load_opt_v2g_glob, elec_price)
    print("##\nCost V2G global = %.2f\n##" % cost_opt_v2g_glob)
          
    return ev_load_opt_v1g, ev_load_opt_v2g, ev_load_opt_v2g_glob, \
        cost_opt_v1g_dep+cost_opt_v1g_arr, cost_opt_v2g_dep+cost_opt_v2g_arr, \
        cost_opt_v2g_glob

if __name__ == "__main__":
    
    n_ev = 4
    p_max_evs = [3, 3, 22, 22]
    soc_max = 40 * np.ones(n_ev)
    p_max_cs = 40
     
    step_time = 1800
    
    T = 48
    t_dep = [13, 12, 11, 14]
    t_arr = [38, 39, 39, 37]

    init_soc = np.zeros(n_ev) # initial SOC
    dep_soc = 10 * np.ones(n_ev) # SOC needed at departure
    e_drive = 4 * np.ones(n_ev) # E consumed when driving
    
    # Monte-Carlo simulations
    n_draws = 100
    cost_v1g = np.empty(n_draws)
    cost_v2g = np.empty(n_draws)
    cost_v2g_glob = np.empty(n_draws)
    load_v1g, load_v2g, load_v2g_glob = 0, 0, 0
    for i_draw in range(n_draws):
        print("draw %i" % (i_draw+1))
        
        elec_price = np.random.rand(T) # elec. prices
        
        load_v1g, load_v2g, load_v2g_glob, cost_1, cost_2, cost_3 = \
            compare_charging_strat(T, n_ev, t_dep, t_arr, p_max_evs, init_soc, 
                                   dep_soc, p_max_cs, step_time, elec_price)
        cost_v1g[i_draw] = cost_1
        cost_v2g[i_draw] = cost_2
        cost_v2g_glob[i_draw] = cost_3
        
    print("Mean cost V1G = %.2f" % np.mean(cost_v1g))
    print("Mean cost V2G = %.2f" % np.mean(cost_v2g))
    print("Mean cost V2G global = %.2f" % np.mean(cost_v2g_glob))
        
    
