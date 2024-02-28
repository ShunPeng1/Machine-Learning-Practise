import os
import heapq


class PriorityQueue:
  def __init__(self):
    self._queue = []
    self._index = 0

  def push(self, item, priority):
    heapq.heappush(self._queue, (priority, self._index, item))
    self._index += 1

  def pop(self):
    return heapq.heappop(self._queue)[-1]

  def is_empty(self):
    return len(self._queue) == 0

  def find(self, item):
    for _, _, i in self._queue:
      if i == item:
        return True
    return False

  def pop_item(self, item):
        for i in range(len(self._queue)):
            if self._queue[i][-1] == item:
                return self._queue.pop(i)[-1]
        return None

  def get_priority(self, item):
        for priority, _, i in self._queue:
            if i == item:
                return -priority
        return None

  def __len__(self):
        return len(self._queue)
  


# Directed, weighted graphs
class Graph:
  def __init__(self):
    self.AL = dict() # adjacency list
    self.V = 0
    self.E = 0

  def __str__(self):
    res = 'V: %d, E: %d\n'%(self.V, self.E)
    for u, neighbors in self.AL.items():
      line = '%d: %s\n'%(u, str(neighbors))
      res += line
    return res

  def print(self):
    print(str(self))

  def load_from_file(self, filename):
    '''
        Example input file:
            V E
            u v w
            u v w
            u v w
            ...

        # input.txt
        7 8
        0 1 5
        0 2 6
        1 3 12
        1 4 9
        2 5 5
        3 5 8
        3 6 7
        4 6 4
    '''
    if os.path.exists(filename):
      with open(filename) as g:
        self.V, self.E = [int(it) for it in g.readline().split()]
        for line in g:
          u, v, w = [int(it) for it in line.strip().split()]
          if u not in self.AL:
            self.AL[u] = []
          self.AL[u].append((v, w))


class SearchStrategy:
  def search(self, g: Graph, src: int, dst: int) -> tuple:
    expanded = [] # list of expanded vertices in the traversal order
    path = [] # path from src to dst
    return expanded, path
  

class BFS(SearchStrategy):
  def search(self, g: Graph, src: int, dst: int) -> tuple:
    expanded = [] # list of expanded vertices in the traversal order
    path = [] # path from src to dst

    # TODO 1

    queue = [src]
    precessor = {src : -1}

    while (len(queue) != 0) :

      # traverse
      node = queue.pop(0)
      expanded.append(node)

      if (node == dst):
        break

      if node in g.AL:
        for edge in g.AL[node] :
          #print(node,edge[0],edge[1], expanded)
          succ = edge[0]

          if succ in expanded or succ in queue:
            #print("Expanded")
            continue

          # expand
          queue.append(succ)
          precessor[succ] = node


    # backtracking
    backtrack = dst
    while backtrack != -1 and backtrack in precessor:
      path.append(backtrack)
      backtrack = precessor[backtrack]      
    path.reverse()

    return expanded, path
  

class DFS(SearchStrategy):
  def search(self, g: Graph, src: int, dst: int, limit: int = -1) -> tuple:
    expanded = [] # list of expanded vertices in the traversal order
    path = [] # path from src to dst

    # TODO 2

    stack = [(src, 0)]
    precessor = {src : -1}

    while (len(stack) != 0) :

      # traverse
      node, height = stack.pop()
      expanded.append(node)

      if (node == dst):
        break

      if node in g.AL:
        for edge in g.AL[node] :
          #print(node,edge[0],edge[1], expanded)
          succ = edge[0]

          if succ in expanded or succ in stack or (limit != -1 and height >= limit):
            #print("Expanded")
            continue

          # expand
          stack.append((succ, height+1))
          precessor[succ] = node


    # backtracking
    backtrack = dst
    while backtrack != -1 and backtrack in precessor:
      path.append(backtrack)
      backtrack = precessor[backtrack]      
    path.reverse()



    return expanded, path
  


class UCS(SearchStrategy):
  def search(self, g: Graph, src: int, dst: int) -> tuple:
    expanded = [] # list of expanded vertices in the traversal order
    path = [] # path from src to dst

    # TODO 3
    
    pqueue = PriorityQueue()
    precessor = {src : (-1,0)}

    pqueue.push(src,0)

    while (len(pqueue) != 0) :

      # traverse
      node = pqueue.pop()
      expanded.append(node)

      if (node == dst):
        break

      if node not in g.AL:
        continue

      for edge in g.AL[node] :
        #print(node,edge[0],edge[1], expanded)
        succ, weight = edge
        sumWeight = weight + precessor[node][1]
        
        if succ in expanded:
          #print("Expanded")
          continue

        if pqueue.find(succ):
          #print("Found in pqueue", succ, sumWeight, pqueue.get_priority(succ))
          if sumWeight < pqueue.get_priority(succ):
            pqueue.pop_item(succ)
            pqueue.push(succ, sumWeight)
            precessor[succ] = (node,sumWeight)

        else: 
          #print("Add ", succ, sumWeight)
          pqueue.push(succ, sumWeight)
          precessor[succ] = (node,sumWeight)


    # backtracking
    backtrack = dst
    while backtrack != -1 and backtrack in precessor:
      path.append(backtrack)
      backtrack = precessor[backtrack][0]      
    path.reverse()

    return expanded, path
  

class DLS(SearchStrategy):
  def __init__(self, LIM: int):
    self.LIM = LIM

  def search(self, g: Graph, src: int, dst: int) -> tuple:
    expanded = [] # list of expanded vertices in the traversal order
    path = [] # path from src to dst

    # TODO 4
    expanded, path = DFS().search(g, src, dst, self.LIM)

    return expanded, path
  
class IDS(SearchStrategy):
  def __init__(self, MAX_LIM: int):
    self.MAX_LIM = MAX_LIM

  def search(self, g: Graph, src: int, dst: int) -> tuple:
    expanded = [] # list of expanded vertices in the traversal order
    path = [] # path from src to dst

    # TODO 5
    for i in range(1, self.MAX_LIM+1):
      expanded, path = DLS(LIM=i).search(g, src, dst)
      if len(path) > 0:
        break

    return expanded, path
  
bfs = BFS()
dfs = DFS()
ucs = UCS()
dls = DLS(LIM=3)
ids = IDS(MAX_LIM=5)

g = Graph()

g.load_from_file(r'C:\Users\PC\Downloads\ML\Machine Learning Practise\Search\input.txt')
print("Graph: ")
g.print()

for stg in [bfs, dfs, ucs, dls, ids]:
  print(stg)
  expanded, path = stg.search(g, 0, g.V-1)
  print(expanded)
  print(path)


