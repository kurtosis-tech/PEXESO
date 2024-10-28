import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams

df = pd.read_csv("../result/pexeso_joinable_columns.csv")
G = nx.from_pandas_edgelist(df, source='LeftColumn', target='RightColumn', edge_attr='RelationRatio', create_using=nx.DiGraph())

rcParams['figure.figsize'] = 14, 10
pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
d = dict(G.degree)
nx.draw(G, pos, node_color='lightblue', 
        with_labels=True, node_size=4, font_size=5, width=0.3)

plt.show()
plt.savefig('columns.png')
