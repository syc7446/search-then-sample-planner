import math


class Node:
    def __init__(self, node=None, cut_dim=0, parent=None, left=None, right=None):
        self.node = node
        self.cut_dim = cut_dim
        self.parent = parent
        self.left = left
        self.right = right

    __str__ = lambda self: str(self.node)
    __getitem__ = lambda self, x: self.node[x]


class KDTree:
    def __init__(self, root):
        self.root = Node(node=root)
        self.length = 0
        self.dim = len(root)

    def addNode(self, new):
        cur = self.root
        if not isinstance(new, Node): new = Node(new)

        while True:
            if new[cur.cut_dim] < cur[cur.cut_dim]:
                if cur.left is None:
                    cur.left = new
                    new.parent = cur
                    new.cut_dim = (cur.cut_dim + 1) % self.dim
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = new
                    new.parent = cur
                    new.cut_dim = (cur.cut_dim + 1) % self.dim
                    break
                cur = cur.right
        self.length += 1

    def initDistBetTrees(self, start, goal):
        self.dist_bet_trees = math.hypot(start[0]-goal[0], start[1]-goal[1])
        self.norm_dist_bet_trees = self.dist_bet_trees

    def setDistBetTrees(self, new, other_tree):
        node, dist = other_tree.nearestNode(new)
        if self.dist_bet_trees > dist:
            self.dist_bet_trees = dist
            self._updateDistBetTrees(dist, other_tree)

    def getDistBetTrees(self):
        return self.dist_bet_trees / self.norm_dist_bet_trees

    def _updateDistBetTrees(self, dist, other_tree):
        other_tree.dist_bet_trees = dist

    def nearestNode(self, next, return_node=False):
        next = Node(next)
        leafNode = self._findLeafNode(next, self.root)
        node, dist, _, _ = self._kNearestNodeUp(next, leafNode, None, float('inf'), 0, [], [])
        return node if return_node else tuple(node.node), dist

    def kNearestNode(self, next, k=3):
        next = Node(next)
        leafNode = self._findLeafNode(next, self.root)
        if k > self.length+1: k = self.length+1 # k is given larger than the size of the tree
        if leafNode.node != self.root.node:
            _, _, qNode, _ = self._kNearestNodeUp(next, leafNode, None, float('inf'), k, [], [])
        else:
            _, _, qNode, _ = self._kNearestNodeDown(next, leafNode, None, float('inf'), k, [], [])
        return qNode

    def _findLeafNode(self, next, cur):
        if next[cur.cut_dim] < cur[cur.cut_dim]:
            if cur.left is not None:
                leafNode = self._findLeafNode(next, cur.left)
            else:
                leafNode = self._findLeafNode(next, cur.right) if cur.right is not None else cur
        else:
            if cur.right is not None:
                leafNode = self._findLeafNode(next, cur.right)
            else:
                leafNode = self._findLeafNode(next, cur.left) if cur.left is not None else cur
        return leafNode

    def _kNearestNodeUp(self, next, cur, minNode, minDist, k, qNode, qDist):
        dist = self._distance(next.node, cur.node)
        if dist < minDist:
            minNode, minDist = cur, dist
        if k > 0: qNode, qDist = self._append_q(cur, dist, k, qNode, qDist)

        if cur.parent is not None:
            # Compare with the sibling node
            if cur == cur.parent.left: sibNode = cur.parent.right
            else: sibNode = cur.parent.left
            if sibNode is not None:
                minNode, minDist, qNode, qDist = self._kNearestNodeDown(next, sibNode,
                                                                        minNode, minDist,
                                                                        k, qNode, qDist)

            minNode, minDist, qNode, qDist = self._kNearestNodeUp(next, cur.parent,
                                                                  minNode, minDist,
                                                                  k, qNode, qDist)
        return minNode, minDist, qNode, qDist

    def _kNearestNodeDown(self, next, cur, minNode, minDist, k, qNode, qDist):
        dist = self._distance(next.node, cur.node)
        if minDist > dist:
            minNode, minDist = cur, dist
        if k > 0: qNode, qDist = self._append_q(cur, dist, k, qNode, qDist)

        if next[cur.cut_dim] - cur[cur.cut_dim] > 0:    # right
            if cur.right is not None:
                minNode, minDist, qNode, qDist = self._kNearestNodeDown(next, cur.right,
                                                                        minNode, minDist,
                                                                        k, qNode, qDist)
            if k > 0 and k > len(qNode) and cur.left is not None:
                minNode, minDist, qNode, qDist = self._kNearestNodeDown(next, cur.left,
                                                                        minNode, minDist,
                                                                        k, qNode, qDist)
        else:   # left
            if cur.left is not None:
                minNode, minDist, qNode, qDist = self._kNearestNodeDown(next, cur.left,
                                                                      minNode, minDist,
                                                                      k, qNode, qDist)
            if k > 0 and k > len(qNode) and cur.right is not None:
                minNode, minDist, qNode, qDist = self._kNearestNodeDown(next, cur.right,
                                                                        minNode, minDist,
                                                                        k, qNode, qDist)
        return minNode, minDist, qNode, qDist

    def _distance(self, node_a, node_b):
        sum_square_diff = 0
        for d in range(len(node_a)):
            sum_square_diff += (node_a[d] - node_b[d]) ** 2
        return math.sqrt(sum_square_diff)

    def _append_q(self, cur, dist, k, qNode, qDist):
        if len(qNode) < k:
            qNode.append(cur)
            qDist.append(dist)
        else:
            maxVal = max(qDist)
            if maxVal > dist:
                maxIndex = qDist.index(max(qDist))
                qNode.pop(maxIndex)
                qDist.pop(maxIndex)
                qNode.append(cur)
                qDist.append(dist)
        return qNode, qDist

    def __str__(self):
        self.printTree(self.root, 0)
        return ""

    def printTree(self, node, depth=0, newNode=None):
        if node is None: return
        if newNode is not None:
            print(" | " * depth, node, "%0.4f" % self._distance(newNode.node, node.node))
        else:
            print(" | " * depth, node)
        self.printTree(node.left, depth + 1, newNode)
        self.printTree(node.right, depth + 1, newNode)