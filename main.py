#!/usr/bin/python
# -*- coding: UTF-8 -*-

import random
import copy
import numpy as np
import networkx as nx
import pandas as pd
import os
# networkx version 2.4

# Two choices 1 and 0 for PV and No PV
# Three choices 0，1，2 for PV recycle, 0 for not, 1 for pre-paid for recycle, 2 for plus for reuse

class pv_recycle (object):
	def __init__(self, nodes=1000, degree=3, periods =60, disposal_cost=200,reuse_price=50,scenario=3,PVcost=5200,fit=0,preadopt=0.1,gov_inv_disposal=0,gov_inv_reuse=0,recycleprob=0.1,breakprob=0.005,results_path='./hhh/'):
		# parameters about social network
		self.nodes = nodes
		self.G = nx.Graph()
		self.graph_type = 'BA'
		self.degree = degree
		self.periods = periods
		self.reuse_price=reuse_price
		self.disposal_cost=disposal_cost

		# PV parameters
		self.PV_life = 25
		self.PV_fixcost = PVcost
		self.whole_life_discount = (1-1.08**(-self.PV_life))/(1-1/1.08)*12
		self.monthlyuh = [63.61, 76.47, 109.88, 130.51, 145.50,134.76, 122.37, 111.73, 101.63, 85.59, 62.68, 55.28]

		self.gov_inv_disposal=gov_inv_disposal
		self.gov_inv_reuse=gov_inv_reuse
		self.recycle_service_price = {0: 0, 1: self.disposal_cost-self.gov_inv_disposal, 2: self.disposal_cost+self.reuse_price-self.gov_inv_disposal-self.gov_inv_reuse}
		self.total_gov_invest=0

		self.scenario = scenario
		disposal={1:0,2:self.disposal_cost,3:self.disposal_cost/2}
		recycle_service={1:0,2:1,3:2}
		self.PV_disposal = disposal[self.scenario]
		self.recycle_service = recycle_service[self.scenario]

		self.electricity_price = 0.3598
		self.governmentsubsidy=fit
		self.total_gov_sub=0
		self.stateprofit = 1200/12*(self.electricity_price+self.governmentsubsidy)

		self.monthrealprob = breakprob
		self.recycleprob=recycleprob

		self.all_agent = []
		self.preadopt = preadopt
		self.agents_total_benefits = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_average_benefits = dict(zip(range(0, self.nodes), [0]*self.nodes))

		self.agents_pv_decision = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_recycle_choice = dict(zip(range(0, self.nodes), [0]*self.nodes))

		self.agents_break_statues = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_recycle_statues = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_install_time = {}
		self.agents_expectprob_recycle = {}
		self.agents_realprob_recycle = {}

		# For statistic
		self.totalbreak = [0]
		self.totaladopt = [0]
		self.totalrecycle = [0]

		self.results_path=results_path

	def reset(self):
		self.totalbreak = [0]
		self.totaladopt = [0]
		self.totalrecycle = [0]
		self.total_gov_sub=0
		self.total_gov_invest=0
		self.all_agent = []
		self.agents_total_benefits = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_average_benefits = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_pv_decision = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_recycle_choice = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_break_statues = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_recycle_statues = dict(zip(range(0, self.nodes), [0]*self.nodes))
		self.agents_install_time = {}
		self.agents_expectprob_recycle = {}
		self.agents_realprob_recycle = {}


	def populate(self):
		self.all_agent = list(range(self.nodes))
		for i in self.all_agent[0:int(self.nodes*self.preadopt)]:
			self.agents_pv_decision[i] = 1
			self.agents_install_time[i] = -1
		self.totaladopt.append(int(self.nodes*self.preadopt))

		random.shuffle(self.all_agent)

		random_recycle_prob=np.random.rand(self.nodes)*self.recycleprob*2
		self.agents_expectprob_recycle = dict(zip(range(0, self.nodes),random_recycle_prob))
		self.agents_realprob_recycle = dict(zip(range(0, self.nodes),(1-(1-random_recycle_prob)**(1/60))))

		def bagraph():
			self.G = nx.barabasi_albert_graph(self.nodes, self.degree)

		def ergraph():
			self.G = nx.erdos_renyi_graph(self.nodes, self.degree)

		def wsgraph():
			self.G = nx.watts_strogatz_graph(self.nodes, self.degree, 0.3)

		creategraph = {'BA': bagraph, 'ER': ergraph, 'WS': wsgraph}
		creategraph[self.graph_type]()

	def cal_monthly_benefit(self, node, time):
		tor = 1
		while tor > 0.2 or tor < -0.2:
			tor = np.random.randn()/20
		utilisehour=(1+tor)*self.monthlyuh[time % 12]
		self.total_gov_sub+=utilisehour*(self.governmentsubsidy*(1/1.00643403011000343)**time)
		return utilisehour*(self.electricity_price+self.governmentsubsidy)

	def if_bad_thing_happen(self, node, time):
		b = False
		r = False
		if self.agents_break_statues[node] == 0:
			if np.random.rand() < self.monthrealprob:
				b = True
			else:
				if (self.agents_recycle_statues[node] == 0) and (np.random.rand() < self.agents_realprob_recycle[node]):
					r = True
		return b, r

	def cal_ecpected_average_profits(self, node, time):
		count_income = 0
		count_neighbor_adopters = 0
		count_break=0
		prob=1
		for neighbor in self.G.neighbors(node):
			if self.agents_pv_decision[neighbor] == 1:
				count_income += self.agents_average_benefits[neighbor]
				count_neighbor_adopters += 1
				if self.agents_break_statues[node]==1:
					count_break+=1
		if count_break==0:
			prob=max(1-time/24,0)
		if count_neighbor_adopters == 0:
			return self.stateprofit,1
		elif count_neighbor_adopters > 0:
			return count_income/count_neighbor_adopters,prob

	def cal_best_choice(self, node, time):
		avep,overallbreakratio = self.cal_ecpected_average_profits(node, time)
		totalbreakprob=overallbreakratio+self.agents_expectprob_recycle[node]
		income0 = avep*self.whole_life_discount*((1-totalbreakprob)+totalbreakprob*0.8)-self.PV_disposal*self.recycle_service
		income1 = avep*self.whole_life_discount*((1-totalbreakprob)+totalbreakprob*0.8)-self.recycle_service_price[1]-100000*(2-self.recycle_service)
		income2 = avep*self.whole_life_discount*((1-overallbreakratio)+overallbreakratio*0.8)-self.recycle_service_price[2]-100000*(2-self.recycle_service)
		income = income0
		recy = 0
		if income1 > income0:
			if income2 > income1:
				income = income2
				recy = 2
			else:
				income = income1
				recy = 1
		if income > self.PV_fixcost:
			return 1, recy
		else:
			return 0, 0

	def update(self, time):
		new_total_benefits = copy.deepcopy(self.agents_total_benefits)
		new_average_benefits = copy.deepcopy(self.agents_average_benefits)
		new_pv_decision = copy.deepcopy(self.agents_pv_decision)
		new_recycle_choice = copy.deepcopy(self.agents_recycle_choice)
		new_break_statues = copy.deepcopy(self.agents_break_statues)
		new_recycle_statues = copy.deepcopy(self.agents_recycle_statues)
		new_install_time = copy.deepcopy(self.agents_install_time)

		newbreak = 0
		newadopt = 0
		newrecycle = 0
		for node in self.all_agent:
			if self.agents_pv_decision[node] == 0:
				x,recy = self.cal_best_choice(node, time)
				if x == 1:
					newadopt+=1
					new_pv_decision[node] = 1
					new_install_time[node] = time
					new_recycle_choice[node] = recy
					if recy == 1:
						self.total_gov_invest+=self.gov_inv_disposal**(1/1.00643403011000343)**time
					elif recy ==2:
						self.total_gov_invest+=(self.gov_inv_disposal+self.gov_inv_reuse)**(1/1.00643403011000343)**time
			elif self.agents_pv_decision[node] == 1:
				if self.agents_break_statues[node] == 0 and self.agents_recycle_statues[node] == 0:
					b, r = self.if_bad_thing_happen(node, time)
					if b == True:
						new_break_statues[node] = 1
						newbreak += 1
					elif b == False and r == True:
						new_recycle_statues[node] = 1
						newrecycle += 1
				if self.agents_break_statues[node] == 0 and (self.agents_recycle_statues[node] == 0 or (self.agents_recycle_statues[node] == 1 and self.agents_recycle_choice[node] == 2)):
					new_total_benefits[node] += self.cal_monthly_benefit(node, time)
					new_average_benefits[node] = new_total_benefits[node]/(time-self.agents_install_time[node])

		self.totalbreak.append(self.totalbreak[-1]+newbreak)
		self.totaladopt.append(self.totaladopt[-1]+newadopt)
		self.totalrecycle.append(self.totalrecycle[-1]+newrecycle)

		self.agents_total_benefits=copy.deepcopy(new_total_benefits)
		self.agents_average_benefits=copy.deepcopy(new_average_benefits)
		self.agents_pv_decision=copy.deepcopy(new_pv_decision)
		self.agents_recycle_choice=copy.deepcopy(new_recycle_choice)
		self.agents_break_statues=copy.deepcopy(new_break_statues)
		self.agents_recycle_statues=copy.deepcopy(new_recycle_statues)
		self.agents_install_time=copy.deepcopy(new_install_time)

	def run(self,rounds=100):
		PV_adopter=[]
		PV_recycle1=[]
		PV_recycle2=[]
		gov_fitsub=[]
		gov_recsub=[]
		alpha=0
		while alpha<rounds:
			alpha+=1
			self.reset()
			self.populate()
			for time in range(self.periods):
				self.update(time)
			PV_adopter.append(self.totaladopt)
			x=pd.value_counts(list(self.agents_recycle_choice.values()))
			if 1 in x:
				PV_recycle1.append(x[1])
			else:
				PV_recycle1.append(0)
			if 2 in x:
				PV_recycle2.append(x[2])
			else:
				PV_recycle2.append(0)
			gov_fitsub.append(self.total_gov_sub)
			gov_recsub.append(self.total_gov_invest)
			print(alpha,'   ',self.totaladopt[-1])
		np.savetxt(self.results_path+"adopt.csv",np.array(PV_adopter),fmt='%d',delimiter=",")
		np.savetxt(self.results_path+"recycle.csv",np.array([PV_recycle1,PV_recycle2]),fmt='%d',delimiter=",")
		np.savetxt(self.results_path+"gov.csv",np.array([gov_fitsub,gov_recsub]),fmt='%f',delimiter=",")
		print('average total adoptor:%.2f'%pd.DataFrame(PV_adopter)[61].mean())