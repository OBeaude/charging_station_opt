# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:07 2020

@author: B57876
"""

import numpy as np
import pulp
from pulp import LpVariable, LpAffineExpression, lpSum
# perso
from plot import plot_flex_load, simple_plot

class OptimV2g:
    def __init__(self, is_dep_pb, T, t_dep, t_arr, p_max_evs, init_soc, dep_soc,
                 soc_max, p_max_cs, step_time, elec_price, solver: str="COIN", 
                 time_limit_s: int=60):
        """
        This class solve the first phase of the ev charging problem, by minimizing the impact on the actual pic of
        consumption base on the forecast. After the initialisation of the object you can call solve() to solve it.

        :param T nber of time-slots
        """

        self.__is_dep_pb = is_dep_pb
        self.__T = max(t_dep) if is_dep_pb else T-min(t_arr)
        self.__n_ev = len(t_dep)
        self.__t_dep = t_dep
        self.__t_arr = t_arr
        self.__p_max_evs = p_max_evs          
        self.__init_soc = init_soc
        self.__dep_soc = dep_soc
        self.__soc_max = soc_max                      
        self.__p_max_cs = p_max_cs
        # time-slot duration
        self.delta_t_s = step_time
        self.__elec_price = elec_price[:self.__T] if is_dep_pb \
                                                    else elec_price[-self.__T:]
        self.__solver = solver
        self.__time_limit_s = time_limit_s
       
    def __prob_init(self):
        """
        Initialise the disaggreg. optimization problem
        
        :param n_iter_max_iterative_proc: maximal number of iteration in the
        iterative procedure to be run (to initialize matrices to follow different
        elements during the dynamics)        
        """                                          
                
        # Create pulp Linear problem
        self.__prob = pulp.LpProblem("Smart charging V2G", pulp.LpMinimize)

        self.__define_variables()
        self.__define_objective()
        self.__define_constraints()

    def __define_variables(self):
        """Define the variables of the Linear problem"""

        self.__flex_load = {}
        # loop over EVs to fix bounds 
        for i_ev in range(self.__n_ev):
            self.__flex_load[i_ev] = LpVariable.dicts("p_ve_%i" % i_ev,
                                                      indexs=np.arange(self.__T),
                                                      lowBound=-self.__p_max_evs[i_ev],
                                                      upBound=self.__p_max_evs[i_ev]
                                                      )
        
    def __define_objective(self):
        """Define the objective of the Linear problem"""
        objective = LpAffineExpression()
        
        for i_ev in range(self.__n_ev):
            for t in range(self.__T):
                objective.addterm(self.__flex_load[i_ev][t], self.__elec_price[t])
                
        self.__prob += lpSum(objective)

    def __define_constraints(self):
        """Define the constraints of the Linear problem"""
        # 1.4.1) Total E constraint
        for i_ev in range(self.__n_ev):
            self.__prob += lpSum([self.__flex_load[i_ev][t] \
                                        for t in np.arange(self.__T)]) \
              >= 3600/self.delta_t_s * (self.__dep_soc[i_ev]-self.__init_soc[i_ev]), \
                                      "Contrainte energie EV%i" % (i_ev+1)

        # 1.4.2) Upper-bounds as constraints
        for t in range(self.__T):
            for i_ev in range(self.__n_ev):
                # impose zero-power if EV not plugged-in
                if self.__is_dep_pb and t>=self.__t_dep[i_ev]:
                    self.__prob += self.__flex_load[i_ev][t] == 0, \
                      "Zero-power when not plugged-in_t%i_EV%i" % (t+1, i_ev+1)
                elif not self.__is_dep_pb and t<self.__t_arr[i_ev]-min(self.__t_arr):
                    self.__prob += self.__flex_load[i_ev][t] == 0, \
                      "Zero-power when not plugged-in_t%i_EV%i" % (t+1, i_ev+1)                      
                else:
                    self.__prob += self.delta_t_s / 3600 * lpSum([self.__flex_load[i_ev][s] \
                                        for s in np.arange(t+1)]) \
              >= -self.__init_soc[i_ev], "Contrainte positive SOC_t%i_EV%i" % (t+1,i_ev+1)
                    self.__prob += self.delta_t_s / 3600 * lpSum([self.__flex_load[i_ev][s] \
                                        for s in np.arange(t+1)]) \
              <= self.__soc_max[i_ev]-self.__init_soc[i_ev], \
              "Contrainte capacite SOC_t%i_EV%i" % (t+1,i_ev+1)

            # Aggreg. constraint
            self.__prob += lpSum([self.__flex_load[i_ev][t] \
                                        for i_ev in np.arange(self.__n_ev)]) \
              <= self.__p_max_cs, "Contrainte pos profil aggreg_t%i" % (t+1)
            self.__prob += lpSum([self.__flex_load[i_ev][t] \
                                        for i_ev in np.arange(self.__n_ev)]) \
              >= -self.__p_max_cs, "Contrainte neg profil aggreg_t%i" % (t+1)

    def solve(self, directory_output=None):
        """

        :return Solution object
        :exception NoSolutionFoundException
        :exception ....


        ........


        OUTPUT:
        * optim_status: optimization status, given by used solver
        * flex_load_opt: optimal consumption profile of flexible load (of size T)
        """

        # Init. pb
        self.__prob_init()
# DEBUG
        print("Start solving Smart Charging V2G as an LP problem")
        # 1.5) The problem data is written to an .lp file (TBDiscussed: necessary?)
        # directory where .lp file will be saved
        if directory_output:
            output_filename = "%s/" % directory_output
        else:
            output_filename = ""
        # complete .lp filename
        output_filename += "optim_v2g_dep_n_ev%i.lp" % self.__n_ev if self.__is_dep_pb \
                                    else "optim_v2g_arr_n_ev%i.lp" % self.__n_ev
        # write .lp file
        self.__prob.writeLP(output_filename)
# DEBUG
        print("LP problem written in %s" % output_filename)
        
        # 2) The problem is solved using COIN or PuLP's default choice if COIN is not available
        try:
            if self.__solver == "GUROBI": # if GUROBI
                self.__prob.solve(pulp.solvers.GUROBI(msg=0, Method=2,
                                                      Presolve=2, RINS=500,
                                                      TimeLimit=self.__time_limit_s))
            else:
                self.__prob.solve(pulp.COIN_CMD())  # Try with COIN...
        except pulp.PulpSolverError:       # ... if not installed take default solver
# DEBUG
            print("COIN-OR not installed try to use default solver of the system")
            self.__prob.solve(pulp.PULP_CBC_CMD(maxSeconds=self.__time_limit_s))
    
        # 3) Get the optimal values for the variables of the optim. pb
        if self.__prob.status is not pulp.LpStatusOptimal:
            # no solution found, 
# DEBUG
            print("Smart Charging V2G pb not solved to optimality")
            
            flex_load_opt = -1
        else:
            flex_load_opt = np.empty([self.__n_ev, self.__T])
            for i_ev in range(self.__n_ev):
                flex_load_opt[i_ev,:] = np.array([self.__flex_load[i_ev][t].varValue \
                                            for t in np.arange(self.__T)])
# INFO
            print("LP disaggreg. problem solved")

                 
        return flex_load_opt

def calc_load_cost(ev_load, elec_price):
    
    return sum(np.sum(ev_load, axis=0)*elec_price)

# Test
if __name__ == '__main__':

    p_max_evs = [3, 3, 22, 22]
    soc_max = 40 * np.ones(4)
    p_max_cs = 40
    T = 48
    step_time = 1800
    elec_price = np.random.rand(T)
    solver = "COIN"
    time_limit_s = 60
    
    # first phase - before departure
    print("before departure")
    is_dep_pb = True
    t_dep = [13, 12, 11, 14]
    t_arr = np.zeros(4)
    init_soc = np.zeros(4)
    dep_soc = 10 * np.ones(4)
    pb_v2g_dep = OptimV2g(is_dep_pb, T, t_dep, t_arr, p_max_evs, init_soc, dep_soc,
                          soc_max, p_max_cs, step_time, elec_price, solver, 
                          time_limit_s)
    ev_load_opt_dep = pb_v2g_dep.solve()
    
    cost_opt_dep = calc_load_cost(ev_load_opt_dep, elec_price[:ev_load_opt_dep.shape[1]])
    print("elec price = ", elec_price[:ev_load_opt_dep.shape[1]])
    print("opt EV load dep = ", ev_load_opt_dep)
    print("cost_opt = ", cost_opt_dep)
    
    # second phase - after arrival
    print("after arrival")
    is_dep_pb = False
    t_dep = T*np.ones(4)
    t_arr = [38, 39, 39, 37]
    init_soc = init_soc + np.sum(ev_load_opt_dep, axis=1) - 4
    print("init_soc = ", init_soc)
    dep_soc = np.zeros(4)
    pb_v2g_arr = OptimV2g(is_dep_pb, T, t_dep, t_arr, p_max_evs, init_soc, dep_soc,
                          soc_max, p_max_cs, step_time, elec_price, solver, 
                          time_limit_s)
    ev_load_opt_arr = pb_v2g_arr.solve()

    cost_opt_arr = calc_load_cost(ev_load_opt_arr, elec_price[-ev_load_opt_arr.shape[1]:])
    print("elec price = ", elec_price[-ev_load_opt_arr.shape[1]:])
    print("opt EV load arr = ", ev_load_opt_arr)
    print("cost_opt = ", cost_opt_arr)
    
    # concatenate results
    ev_load_opt = np.zeros([4,T])
    ev_load_opt[:,:ev_load_opt_dep.shape[1]] = ev_load_opt_dep
    ev_load_opt[:,-ev_load_opt_arr.shape[1]:] = ev_load_opt_arr
    
    plot_flex_load(ev_load_opt, "Time", "Load (kW)", 1, True, "ev_load_v2g")
    simple_plot(np.sum(ev_load_opt, axis=0), "Time", "Load (kW)", 2, True,
                "agg_ev_load_v2g")



