# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:44:07 2020

@author: B57876
"""

import numpy as np
import pulp
from pulp import LpVariable, LpAffineExpression, lpSum

class OptimV2gGlobal:
    def __init__(self, T, t_dep, t_arr, p_max_evs, init_soc, dep_soc, e_drive,
                 soc_max, p_max_cs, step_time, elec_price, solver: str="COIN", 
                 time_limit_s: int=60):
        """
        This class solve the first phase of the ev charging problem, by minimizing the impact on the actual pic of
        consumption base on the forecast. After the initialisation of the object you can call solve() to solve it.

        :param T nber of time-slots
        """

        self.__T = T
        self.__n_ev = len(t_dep)
        self.__t_dep = t_dep
        self.__t_arr = t_arr
        self.__p_max_evs = p_max_evs          
        self.__init_soc = init_soc
        self.__dep_soc = dep_soc
        self.__e_drive = e_drive
        self.__soc_max = soc_max                      
        self.__p_max_cs = p_max_cs
        # time-slot duration
        self.delta_t_s = step_time
        self.__elec_price = elec_price
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
        self.__prob = pulp.LpProblem("Smart charging V2G global", pulp.LpMinimize)

        self.__define_variables()
        self.__define_objective()
        self.__define_constraints()
        
        for i_ev in range(self.__n_ev):
            if self.__e_drive[i_ev]*3600/self.delta_t_s \
                                    /(self.__t_arr[i_ev]-self.__t_dep[i_ev]) \
                > self.__p_max_evs[i_ev]:
                print("pmax EV_%i trop faible pour assurer la modelisation du roulage" \
                                                                  % (i_ev+1))

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
        # Total E constraint at departure
        for i_ev in range(self.__n_ev):
            self.__prob += lpSum([self.__flex_load[i_ev][t] \
                                        for t in np.arange(self.__t_dep[i_ev])]) \
              >= 3600/self.delta_t_s * (self.__dep_soc[i_ev]-self.__init_soc[i_ev]), \
                                      "Contrainte energie EV%i" % (i_ev+1)
        for t in range(self.__T):
            for i_ev in range(self.__n_ev):
                # impose driving consumption when EV not plugged in
                if t>=self.__t_dep[i_ev] and t<self.__t_arr[i_ev]:
                    self.__prob += self.__flex_load[i_ev][t] == \
                    -self.__e_drive[i_ev]*3600/self.delta_t_s \
                                    /(self.__t_arr[i_ev]-self.__t_dep[i_ev]), \
                      "Contrainte E roulage_t%i_EV%i" % (t+1, i_ev+1)
                
                # constraint on SOC dynamics
                self.__prob += self.delta_t_s / 3600 * lpSum([self.__flex_load[i_ev][s] \
                                    for s in np.arange(t+1)]) \
                                  >= -self.__init_soc[i_ev], \
                                  "Contrainte positive SOC_t%i_EV%i" % (t+1,i_ev+1)
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
        output_filename += "optim_v2g_glob_n_ev%i.lp" % self.__n_ev 
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
            print("Smart Charging V2G global pb not solved to optimality, status = %s" \
                  % self.__prob.status)
            
            flex_load_opt = -1
        else:
            flex_load_opt = np.empty([self.__n_ev, self.__T])
            for i_ev in range(self.__n_ev):
                flex_load_opt[i_ev,:] = np.array([self.__flex_load[i_ev][t].varValue \
                                            for t in np.arange(self.__T)])
                # replace values representing the driving consumption
                for t in range(self.__t_dep[i_ev], self.__t_arr[i_ev]):
                    flex_load_opt[i_ev,t] = 0
# INFO
            print("Smart Charging V2G global problem solved")

                 
        return flex_load_opt

def calc_load_cost(ev_load, elec_price):
    
    return sum(np.sum(ev_load, axis=0)*elec_price)

# Test
if __name__ == '__main__':

    p_max_evs = [3, 3, 22, 22]
    soc_max = 40 * np.ones(4)
    p_max_cs = 40
     
    step_time = 1800
    solver = "COIN"
    time_limit_s = 60
    
    T = 48
    t_dep = [13, 12, 11, 14]
    t_arr = [38, 39, 39, 37]

    elec_price = np.random.rand(T)
    init_soc = np.zeros(4)
    dep_soc = 10 * np.ones(4)
    pb_v2g_glob = OptimV2gGlobal(T, t_dep, t_arr, p_max_evs, init_soc, dep_soc,
                                soc_max, p_max_cs, step_time, elec_price, solver, 
                                time_limit_s)
    ev_load_opt_glob = pb_v2g_glob.solve()
    
    cost_opt = calc_load_cost(ev_load_opt_glob, elec_price)
    print("elec price = ", elec_price)
    print("opt EV load glob = ", ev_load_opt_glob)
    print("cost_opt = ", cost_opt)
    
