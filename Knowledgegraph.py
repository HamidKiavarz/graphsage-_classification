import networkx as nx
import numpy as np
import xlrd
from collections import defaultdict
import pandas as pd
# import matplotlib.pyplot as plt
import stellargraph as sg

def get_Adj_matrix():
    file = "D:\Implementation\Data\Amatrix.xlsx"
    df = pd.read_excel(file, sheet_name='A')
    adj_matrix = df.as_matrix()
    return adj_matrix

if __name__ == "__main__":
    ### adjancacy list
    # adj_matrix= [[0,0,0,0,0,0.5],
    #              [0,0,1,1,0,0],
    #              [0,1,0,0,0.5,0],
    #              [0,1,0,0,0.8,0],
    #              [0,0,1,1,0,0],
    #              [1,0,0,0,0,0]]
    # adj_matrix =[[0, 0, 1],
    #              [0, 0, 1],
    #              [1, 1, 0]]
    
    print(nx.__version__)
    A=np.matrix(get_Adj_matrix())
    G = nx.from_numpy_matrix(A)
    print(G.nodes[2])
    # G = nx.parse_adjlist(lines, nodetype = int)
    # plt = nx.draw(G)
    # plt.show()
    # print(G.nodes())
    # print(G.edges())
