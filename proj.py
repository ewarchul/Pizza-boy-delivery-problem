import math

import copy

import networkx as nx
import matplotlib.pyplot as plt
from random import random, randint
import simplejson
from simanneal import Annealer

#sa_handler = open('sa.txt', 'a')
#rw_handler = open('randwalk.txt', 'a')
astar_handler = open('astar.txt', 'a')
astar_c_handler = open('astar_c.txt', 'a')


number_of_nodes = 80
number_of_edges = 4
max_distance = 1000
h = nx.read_graphml('graph')
g = nx.convert_node_labels_to_integers(h)

alfa = 1.0
beta = 1.0
gamma = 1.0

edges_time = nx.get_edge_attributes(g, 'time')
edges_distance = nx.get_edge_attributes(g, 'distance')
for edge in g.edges:
    g[edge[0]][edge[1]]['objective_frag'] = alfa * edges_distance[(edge[0], edge[1])] + beta * edges_time[(edge[0], edge[1])]

pos = {}
for i in range(len(g.nodes)):
    pos[i] = [g.nodes[i]['x'], g.nodes[i]['y']]
#print(pos)

f = open('route', 'r')
x = simplejson.load(f)
f.close()

x2 = copy.deepcopy(x)
x = [0] + x + [0]
x1 = copy.deepcopy(x)

shortest_paths = dict(nx.shortest_path(g, source=None, target=None, weight='objective_frag'))
shortest_paths_length = dict(nx.shortest_path_length(g, source=None, target=None, weight='objective_frag'))


class PizzaDeliveryProblemAnnealing(Annealer):
    def move(self):
        """Swaps two cities in the route."""
        a = randint(1, len(self.state) - 2)
        b = randint(1, len(self.state) - 2)
        self.state[a], self.state[b] = self.state[b], self.state[a]

    def energy(self):
        e = 0
        time = 0
        edges_time = nx.get_edge_attributes(g, 'time')
        for i in range(len(self.state) - 1):
            current_time = 0
            path = shortest_paths[self.state[i]][self.state[i+1]]
            for i in range(len(path) - 1):
                lower = min(path[i], path[i+1])
                higher = max(path[i], path[i+1])
                current_time += edges_time[(lower, higher)]
            time += current_time
            e += shortest_paths_length[self.state[i]][self.state[i + 1]] + gamma * time**2
        return e


initial_state = x
pizza = tsp = PizzaDeliveryProblemAnnealing(initial_state)
itinerary, miles = tsp.anneal()

#print("Dla symulowanego wyzarzania:")
#print(itinerary)
#print(miles)
#sa_handler.write('{}\n'.format(miles))
#sa_handler.close()
labels = nx.get_edge_attributes(g, 'objective_frag')

number_of_iterations = 50000
e_max = float("inf")
itinerary_max = []
time = 0

for i in range(number_of_iterations):
    a = randint(1, len(x1) - 2)
    b = randint(1, len(x1) - 2)
    x1[a], x1[b] = x1[b], x1[a]
    e = 0
    for k in range(len(x1) - 1):
        current_time = 0
        path = shortest_paths[x1[k]][x1[k + 1]]
        for j in range(len(path) - 1):
            lower = min(path[j], path[j + 1])
            higher = max(path[j], path[j + 1])
            current_time += edges_time[(lower, higher)]
        time += current_time
        e += shortest_paths_length[x1[k]][x1[k + 1]] + gamma * time**2
    if e < e_max:
        e_max = e
        itinerary_max = copy.deepcopy(x1)

#print("Dla bladzenia przypadkowego:")
#print(itinerary_max)
#print(e_max)
#exit()
#rw_handler.write('{}\n'.format(e_max))
#rw_handler.close()
#exit()
heuristic_set = []
heuristic_dictionary = {}
x2.append(0)
for i in x2:
    heuristic_dictionary[i] = {}
for i in x2:
    for k in x2:
        if i != k:
            heuristic_set.append(math.sqrt((g.nodes[i]['x'] - g.nodes[k]['x']) ** 2
                                           + (g.nodes[i]['y'] - g.nodes[k]['y']) ** 2))
            heuristic_dictionary[i].update({k: math.sqrt((g.nodes[i]['x'] - g.nodes[k]['x']) ** 2
                                                         + (g.nodes[i]['y'] - g.nodes[k]['y']) ** 2)})
            heuristic_dictionary[k].update({i: math.sqrt((g.nodes[i]['x'] - g.nodes[k]['x']) ** 2
                                                         + (g.nodes[i]['y'] - g.nodes[k]['y']) ** 2)})
x2.remove(0)

heuristic_set = sorted(heuristic_set)
shortest_distances_heuristic = heuristic_set[0:len(x2)]
heuristic_set = heuristic_set[len(x2):len(heuristic_set)]


path = [0]
cost_sum = 0
cost_min = float("inf")
new_el = 0
last_el = 0
cost = 0
current_heuristic_value = 0
heuristic = 0
length = len(x2)
time = 0

for k in range(length):
    cost_min = float("inf")
    for i in x2:
        cost = shortest_paths_length[last_el][i]
        if heuristic_dictionary[last_el][i] in shortest_distances_heuristic:
            heuristic += sum(shortest_distances_heuristic) - heuristic_dictionary[last_el][i]
        else:
            heuristic += sum(shortest_distances_heuristic) - \
                         shortest_distances_heuristic[len(shortest_distances_heuristic) - 1]
        if cost + heuristic < cost_min:
            cost_min = cost
            new_el = i
            current_heuristic_value = heuristic_dictionary[last_el][new_el]
    current_time = 0
    partial_path = shortest_paths[last_el][i]
    for j in range(len(partial_path) - 1):
        lower = min(partial_path[j], partial_path[j + 1])
        higher = max(partial_path[j], partial_path[j + 1])
        current_time += edges_time[(lower, higher)]
    time += current_time
    path.append(new_el)
    last_el = new_el
    cost_sum += cost_min + gamma * time**2
    astar_c_handler.write('{}\n'.format(cost_sum))
    x2.remove(new_el)
    if current_heuristic_value in shortest_distances_heuristic:
        shortest_distances_heuristic.remove(current_heuristic_value)
    else:
        shortest_distances_heuristic.remove(shortest_distances_heuristic[len(shortest_distances_heuristic) - 1])
cost_sum += shortest_paths_length[last_el][0]
path.append(0)
#print("Dla A*: ")
#print(path)
#print(cost_sum)
astar_c_handler.close()
astar_handler.write('{}\n'.format(cost_sum))
astar_handler.close()
plt.subplot(111)
# edge_labels=nx.draw_networkx_edge_labels(g,pos=nx.spring_layout(g), edge_labels = labels)
nx.draw(g, pos, with_labels=True, font_weight='bold')
#plt.show()
