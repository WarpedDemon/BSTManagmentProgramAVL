import os  # O(1) - Import statement execution.
import random  # O(1) - Import statement execution.

class TreeNode:
    def __init__(self, element):
        self.element = element  # O(1) - Assigns the element to the node.
        self.left = None  # O(1) - Initializes left child as None.
        self.right = None  # O(1) - Initializes right child as None.
        self.height = 1  # O(1) - Initializes the height of the node.

class AVLTree:
    class TreeNode:
        def __init__(self, element):
            self.element = element  # O(1) - Assigns the element to the node.
            self.left = None  # O(1) - Initializes left child as None.
            self.right = None  # O(1) - Initializes right child as None.
            self.height = 1  # O(1) - Initializes the height of the node.

    def __init__(self):
        self.root = None  # O(1) - Initializes the root as None.
        self.size = 0  # O(1) - Initializes the size of the tree as zero.

    def getHeight(self, node):
        if node is None:  # O(1) - Check if the node is None.
            return 0  # O(1) - Return height 0 for None node.
        return node.height  # O(1) - Return the height of the node.

    def updateHeight(self, node):
        if node is not None:  # O(1) - Check if the node is not None.
            # O(log n) - Calculate the height of the node based on the height of its children.
            node.height = max(self.getHeight(node.left), self.getHeight(node.right)) + 1

    def getBalance(self, node):
        if node is None:  # O(1) - Check if the node is None.
            return 0  # O(1) - Return balance factor 0 for None node.
        # O(log n) - Calculate and return the balance factor of the node.
        return self.getHeight(node.left) - self.getHeight(node.right)

    def rotateLeft(self, z):
        y = z.right  # O(1) - Assign right child of z to y.
        T2 = y.left  # O(1) - Assign left child of y to T2.
        y.left = z  # O(1) - Make z the left child of y.
        z.right = T2  # O(1) - Connect T2 as the right child of z.

        # Update heights
        self.updateHeight(z)  # O(log n) - Update the height of z after rotation.
        self.updateHeight(y)  # O(log n) - Update the height of y after rotation.

        return y  # O(1) - Return new root of the subtree.

    def rotateRight(self, z):
        y = z.left  # O(1) - Assign left child of z to y.
        T3 = y.right  # O(1) - Assign right child of y to T3.
        y.right = z  # O(1) - Make z the right child of y.
        z.left = T3  # O(1) - Connect T3 as the left child of z.

        # Update heights
        self.updateHeight(z)  # O(log n) - Update the height of z after rotation.
        self.updateHeight(y)  # O(log n) - Update the height of y after rotation.

        return y  # O(1) - Return new root of the subtree.

    def rebalance(self, node):
        if node is None:  # O(1) - Check if node is None.
            return node  # O(1) - Return None if node is None.
        
        self.updateHeight(node)  # O(log n) - Update the height of the node.
        balance = self.getBalance(node)  # O(log n) - Get the balance factor of the node.

        # Left Left Case
        if balance > 1 and self.getBalance(node.left) >= 0:  # O(log n) - Check for Left Left case.
            return self.rotateRight(node)  # O(log n) - Perform right rotation.

        # Left Right Case
        if balance > 1 and self.getBalance(node.left) < 0:  # O(log n) - Check for Left Right case.
            node.left = self.rotateLeft(node.left)  # O(log n) - Perform left rotation on left child.
            return self.rotateRight(node)  # O(log n) - Perform right rotation.

        # Right Right Case
        if balance < -1 and self.getBalance(node.right) <= 0:  # O(log n) - Check for Right Right case.
            return self.rotateLeft(node)  # O(log n) - Perform left rotation.

        # Right Left Case
        if balance < -1 and self.getBalance(node.right) > 0:  # O(log n) - Check for Right Left case.
            node.right = self.rotateRight(node.right)  # O(log n) - Perform right rotation on right child.
            return self.rotateLeft(node)  # O(log n) - Perform left rotation.

        return node  # O(1) - Return the node if no rotations are needed.

    # Return True if the element is in the tree
    def search(self, e):
        current = self.root  # O(1) - Start from the root.
        while current is not None:  # O(log n) - Traverse the tree until a leaf is reached or element is found.
            if e < current.element:  # O(1) - Check if the element is less than current element.
                current = current.left  # O(1) - Move to the left child.
            elif e > current.element:  # O(1) - Check if the element is greater than current element.
                current = current.right  # O(1) - Move to the right child.
            else:  # O(1) - Element matches current.element.
                return True  # O(1) - Element is found.
        return False  # O(1) - Return False if element is not found.

    # Count and return number of steps taken to find the element
    def countSearch(self, e):
        current = self.root  # O(1) - Start from the root.
        count = 0  # O(1) - Initialize step count.
        while current is not None:  # O(log n) - Traverse the tree until a leaf is reached or element is found.
            if e < current.element:  # O(1) - Check if the element is less than current element.
                current = current.left  # O(1) - Move to the left child.
                count += 1  # O(1) - Increment step count.
            elif e > current.element:  # O(1) - Check if the element is greater than current element.
                current = current.right  # O(1) - Move to the right child.
                count += 1  # O(1) - Increment step count.
            else:  # O(1) - Element matches current.element.
                print(count)  # O(1) - Output the count.
                return count  # O(1) - Return the count.
        print(count)  # O(1) - Output the count.
        return count  # O(1) - Return the count.

    # Insert element e into the AVL tree
    def insert(self, e):
        if self.root is None:  # O(1) - Check if the root is None.
            self.root = self.TreeNode(e)  # O(1) - Create a new node and assign it as the root.
        else:
            self.root = self._insert(self.root, e)  # O(log n) - Recursively insert the element.
        self.size += 1  # O(1) - Increment the size of the tree.

    def _insert(self, node, e):
        if node is None:  # O(1) - Check if the node is None.
            return self.TreeNode(e)  # O(1) - Create and return a new node.
        elif e < node.element:  # O(1) - Compare the element with the node's element.
            node.left = self._insert(node.left, e)  # O(log n) - Recursive call to insert on the left subtree.
        elif e > node.element:  # O(1) - Compare the element with the node's element.
            node.right = self._insert(node.right, e)  # O(log n) - Recursive call to insert on the right subtree.
        else:
            return node  # O(1) - Return the node if it is a duplicate.

        # Update the height of the ancestor node
        self.updateHeight(node)  # O(log n) - Update the height of the node.

        # Rebalance the node if necessary
        return self.rebalance(node)  # O(log n) - Rebalance the tree and return the node.

    def createNewNode(self, e):
        return self.TreeNode(e)  # O(1) - Create and return a new node.

    # Inorder traversal for leaf nodes from the root
    def leaf_inorder(self):
        self.leaf_inorderHelper(self.root)  # O(n) - Call helper function to perform inorder traversal.

    # Helper for inorder traversal for leaf nodes from a subtree
    def leaf_inorderHelper(self, r):
        if r:  # O(1) - Check if the subtree root is not None.
            self.leaf_inorderHelper(r.left)  # O(n) - Recursively traverse the left subtree.
            # Check if node is a leaf node (no children)
            if r.left is None and r.right is None:  # O(1) - Check if both children are None.
                print(r.element, end=" ")  # O(1) - Print the element of the leaf node.
            self.leaf_inorderHelper(r.right)  # O(n) - Recursively traverse the right subtree.

    # Inorder traversal for non-leaf nodes from the root
    def non_leaf_inorder(self):
        self.non_leaf_inorderHelper(self.root)  # O(n) - Call helper function to perform inorder traversal.

    # Helper for inorder traversal for non-leaf nodes from a subtree
    def non_leaf_inorderHelper(self, r):
        if r:  # O(1) - Check if the subtree root is not None.
            self.non_leaf_inorderHelper(r.left)  # O(n) - Recursively traverse the left subtree.
            # Check if node is not a leaf node (has at least one child)
            if r.left is not None or r.right is not None:  # O(1) - Check if at least one child exists.
                print(r.element, end=" ")  # O(1) - Print the element of the non-leaf node.
            self.non_leaf_inorderHelper(r.right)  # O(n) - Recursively traverse the right subtree.

    # Inorder traversal from the root
    def inorder(self):
        self.inorderHelper(self.root)  # O(n) - Call helper function to perform inorder traversal.

    # Inorder traversal from a subtree
    def inorderHelper(self, r):
        if r is not None:  # O(1) - Check if the subtree root is not None.
            self.inorderHelper(r.left)  # O(n) - Recursively traverse the left subtree.
            print(r.element, end=" ")  # O(1) - Print the element.
            self.inorderHelper(r.right)  # O(n) - Recursively traverse the right subtree.

    # Inorder traversal from a subtree that returns a list of elements
    def inorderHelperReturner(self, r):
        ReturnList = []  # O(1) - Initialize an empty list to collect elements.
        if r is not None:  # O(1) - Check if the subtree root is not None.
            ReturnList.extend(self.inorderHelperReturner(r.left))  # O(n) - Recursively collect elements from left subtree.
            ReturnList.append(r.element)  # O(1) - Append the current element.
            ReturnList.extend(self.inorderHelperReturner(r.right))  # O(n) - Recursively collect elements from right subtree.
        return ReturnList  # O(1) - Return the list of collected elements.

    # Inverse Inorder traversal from the root
    def inverse_inorder(self):
        self.inverse_inorderHelper(self.root)  # O(n) - Call helper function to perform inverse inorder traversal.

    # Inverse Inorder traversal from a subtree
    def inverse_inorderHelper(self, r):
        if r is not None:  # O(1) - Check if the subtree root is not None.
            self.inverse_inorderHelper(r.right)  # O(n) - Recursively traverse the right subtree.
            print(r.element, end=" ")  # O(1) - Print the element.
            self.inverse_inorderHelper(r.left)  # O(n) - Recursively traverse the left subtree.

    # Postorder traversal from the root
    def postorder(self):
        self.postorderHelper(self.root)  # O(n) - Call helper function to perform postorder traversal.

    # Postorder traversal from a subtree
    def postorderHelper(self, root):
        if root is not None:  # O(1) - Check if the subtree root is not None.
            self.postorderHelper(root.left)  # O(n) - Recursively traverse the left subtree.
            self.postorderHelper(root.right)  # O(n) - Recursively traverse the right subtree.
            print(root.element, end=" ")  # O(1) - Print the element after its children.

    # Preorder traversal from the root
    def preorder(self):
        self.preorderHelper(self.root)  # O(n) - Call helper function to perform preorder traversal.

    # Preorder traversal from a subtree
    def preorderHelper(self, root):
        if root is not None:  # O(1) - Check if the current node is not None.
            print(root.element, end=" ")  # O(1) - Print the current node's element.
            self.preorderHelper(root.left)  # O(n) - Recursively traverse the left subtree.
            self.preorderHelper(root.right)  # O(n) - Recursively traverse the right subtree.

    # Method to initiate the total preorder traversal from the root
    def total_preorder(self):
        if self.root is None:  # O(1) - Check if the tree is empty.
            print("The tree is empty.")  # O(1) - Output that the tree is empty.
            return
        count = self.total_nodes_preorderHelper(self.root)  # O(n) - Calculate the total number of nodes using preorder traversal.
        print("\n\nTotal nodes in the subtree:", count)  # O(1) - Print the total count of nodes.

    # Helper function for total preorder traversal from a subtree
    def total_nodes_preorderHelper(self, root):
        if root is None:  # O(1) - Check if the current node is None.
            return 0  # O(1) - Return 0 if the subtree is empty.
        print(root.element, end=" ")  # O(1) - Print the current node's element.
        left_count = self.total_nodes_preorderHelper(root.left)  # O(n/2) - Recursively count nodes in the left subtree.
        right_count = self.total_nodes_preorderHelper(root.right)  # O(n/2) - Recursively count nodes in the right subtree.
        return 1 + left_count + right_count  # O(1) - Return the count of nodes including the current node.

    # Method to initiate depth calculation of the subtree rooted at a given node
    def depthSubtreeBST(self, e):
        current = self.root  # O(1) - Start from the root.
        while current:  # O(log n) - Search for the node in a BST has logarithmic complexity on average.
            if e < current.element:  # O(1) - Compare the given element with the current node's element.
                current = current.left  # O(1) - Move to the left child.
            elif e > current.element:  # O(1) - Compare the given element with the current node's element.
                current = current.right  # O(1) - Move to the right child.
            else:
                # Element is found, calculate depth
                depth = self.depthSubtreeBSTHelper(current)  # O(n) - Calculate the depth starting from the found node.
                print(depth)  # O(1) - Print the calculated depth.
                return
        print("ERROR: Subtree rooted at node <N> not found!")  # O(1) - Print error if the element is not found in the tree.

    # Helper method to calculate the depth of the subtree in a post-order style traversal
    def depthSubtreeBSTHelper(self, root):
        if root is None:  # O(1) - Check if the current node is None.
            return -1  # O(1) - Return -1 if the subtree is empty.
        left_depth = self.depthSubtreeBSTHelper(root.left)  # O(n/2) - Calculate the depth of the left subtree.
        right_depth = self.depthSubtreeBSTHelper(root.right)  # O(n/2) - Calculate the depth of the right subtree.
        return 1 + max(left_depth, right_depth)  # O(1) - Return the maximum depth plus one for the current node.

    def deleteRec(self, root, key):
        if root is None:  # O(1) - Check if the current node is None.
            return root, False  # O(1) - Return None and False indicating no deletion occurred.

        deleted = False  # O(1) - Initialize the deleted flag as False.
        if key < root.element:  # O(1) - Compare the key with the current node's element.
            root.left, deleted = self.deleteRec(root.left, key)  # O(log n) - Recursively attempt to delete in the left subtree.
        elif key > root.element:  # O(1) - Compare the key with the current node's element.
            root.right, deleted = self.deleteRec(root.right, key)  # O(log n) - Recursively attempt to delete in the right subtree.
        else:
            # Node with only one child or no child
            deleted = True  # O(1) - Set deleted flag to True because a match was found.
            if root.left is None:  # O(1) - Check if the left child does not exist.
                return root.right, deleted  # O(1) - Return the right child and deleted flag (deleting the current node).
            elif root.right is None:  # O(1) - Check if the right child does not exist.
                return root.left, deleted  # O(1) - Return the left child and deleted flag (deleting the current node).

            # Node with two children, get the inorder successor (smallest in the right subtree)
            temp = self.min_value_node(root.right)  # O(log n) - Find the minimum value node in the right subtree.
            root.element = temp.element  # O(1) - Replace current node's element with inorder successor's element.
            root.right, _ = self.deleteRec(root.right, temp.element)  # O(log n) - Delete the inorder successor.

        # Only attempt to rebalance if a deletion occurred
        if deleted:  # O(1) - Check if deletion occurred.
            root = self.rebalance(root)  # O(log n) - Rebalance the tree at this node.

        return root, deleted  # O(1) - Return the potentially new root and deletion status.

    def min_value_node(self, node):
        current = node  # O(1) - Start at the given node.
        while current and current.left is not None:  # O(log n) - Find the minimum value node in a subtree.
            current = current.left  # O(1) - Move to the left child.
        return current  # O(1) - Return the node with the minimum value.

    # Rebalancing, rotation methods, and other AVL methods should be here as well

    def isEmpty(self):
        return self.size == 0  # O(1) - Return True if the size of the tree is zero, indicating the tree is empty.

    def clear(self):
        self.root = None  # O(1) - Set the root to None.
        self.size = 0  # O(1) - Reset the size to zero.

    def getRoot(self):
        return self.root  # O(1) - Returns the root node of the AVL tree.

    def findNodeDepth(self, e):
        current = self.root  # O(1) - Start at the root of the tree.
        depth = 0  # O(1) - Initialize depth counter.
        while current is not None:  # O(log n) - Loop runs proportionally to the height of the tree.
            if e < current.element:  # O(1) - Compare the search element with the current node's element.
                current = current.left  # O(1) - Move to the left child.
                depth += 1  # O(1) - Increment depth counter.
            elif e > current.element:  # O(1) - Compare the search element with the current node's element.
                current = current.right  # O(1) - Move to the right child.
                depth += 1  # O(1) - Increment depth counter.
            else:
                print(depth)  # O(1) - Print the depth where the element is found.
                return
        print("ERROR: Node <N> not found!")  # O(1) - Print error if the element is not found.

    def insert(self, e):
        if self.root is None:  # O(1) - Check if the tree is empty.
            self.root = self.TreeNode(e)  # O(1) - Insert the new node at the root.
        else:
            self.root = self._insert(self.root, e)  # O(log n) - Insert the new element starting from the root.
        self.size += 1  # O(1) - Increment the size of the tree.
        return True  # O(1) - Indicate that the insertion was successful.

    def _insert(self, node, e):
        if node is None:  # O(1) - Check if we've found the insertion point.
            return self.TreeNode(e)  # O(1) - Create and return a new node.
        elif e < node.element:  # O(1) - Compare the element with the node's element.
            node.left = self._insert(node.left, e)  # O(log n) - Recursive call to insert on the left subtree.
        elif e > node.element:  # O(1) - Compare the element with the node's element.
            node.right = self._insert(node.right, e)  # O(log n) - Recursive call to insert on the right subtree.
        else:
            return node  # O(1) - Return the node itself if it is a duplicate.

        return self.rebalance(node)  # O(log n) - Rebalance the tree after insertion.

    def searchNode(self, e):
        current = self.root  # O(1) - Start at the root.
        while current:  # O(log n) - Loop through nodes proportionally to the height of the tree.
            if e < current.element:  # O(1) - Compare the search element with the current node's element.
                current = current.left  # O(1) - Move to the left child.
            elif e > current.element:  # O(1) - Compare the search element with the current node's element.
                current = current.right  # O(1) - Move to the right child.
            else:
                return current  # O(1) - Return the current node if it matches the search element.
        return None  # O(1) - Return None if the element is not found.

    def createSubtreeBST(self, e):
        node = self.searchNode(e)  # O(log n) - Search for the node containing the element.
        if node is None:  # O(1) - Check if the node is not found.
            print("Element not found in the tree.")  # O(1) - Print error message.
            return None  # O(1) - Return None if the element is not found.

        new_bst = AVLTree()  # O(1) - Create a new instance of AVLTree.
        queue = [node]  # O(1) - Initialize a queue with the found node.
        while queue:  # O(m) - Process each node in the subtree where m is the number of nodes in the subtree.
            current = queue.pop(0)  # O(m) - Remove and get the first element of the queue.
            new_bst.insert(current.element)  # O(log m) - Insert the element into the new subtree.
            if current.left:  # O(1) - Check if left child exists.
                queue.append(current.left)  # O(1) - Append left child to the queue.
            if current.right:  # O(1) - Check if right child exists.
                queue.append(current.right)  # O(1) - Append right child to the queue.
        return new_bst  # O(1) - Return the new AVL tree containing the subtree.

    def DisplayBST(self):
        if self.root:  # O(1) - Check if the tree is not empty.
            lines, _, _, _ = self.display_rec(self.root)  # O(n) - Generate display lines for the tree.
            print("\n")
            print("\t== Binary Tree: shape ==")
            print("\n")
            for line in lines:  # O(n) - Print each line for the visual representation of the tree.
                print('\t', line)
        else:
            print("\t== The tree is empty ==")  # O(1) - Print if the tree is empty.

    def display_rec(self, node):
        if node is None:  # O(1) - Check if the node is None.
            return [], 0, 0, 0  # O(1) - Return empty layout for None node.

        node_label = f"n:{node.element} h:{node.height} b:{self.getBalance(node)}"  # O(1) - Create a label for the node.
        width = len(node_label)  # O(1) - Calculate the width of the label.

        # Base case: the node is a leaf
        if node.left is None and node.right is None:
            return [node_label], width, 1, width // 2  # O(1) - Return the label and layout info for a leaf node.

        # Recursively call on the left child
        if node.right is None:
            lines, n, p, x = self.display_rec(node.left)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + node_label
            second_line = x * ' ' + '/' + (n - x - 1 + len(node_label)) * ' '
            shifted_lines = [line + len(node_label) * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + len(node_label), p + 2, n + len(node_label) // 2

        # Recursively call on the right child
        if node.left is None:
            lines, n, p, x = self.display_rec(node.right)
            first_line = node_label + x * '_' + (n - x) * ' '
            second_line = (len(node_label) + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [len(node_label) * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + len(node_label), p + 2, len(node_label) // 2

        # Node has both children
        left, n, p, x = self.display_rec(node.left)
        right, m, q, y = self.display_rec(node.right)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + node_label + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + len(node_label) + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + len(node_label) * ' ' + b for a, b in zipped_lines]
        return lines, n + m + len(node_label), max(p, q) + 2, n + len(node_label) // 2
    

def Preload():
    Data = []
    RequestLoopCount = input("Please enter the number of elements you would like generated in the array: ")  # O(1) - Input number of elements.
    while int(RequestLoopCount) > 0:
        NewItemToAddToArray = random.randint(0, 100)  # O(1) - Generate a random number.
        Data.append(NewItemToAddToArray)  # O(1) - Append the random number to the data list.
        RequestLoopCount = int(RequestLoopCount) - 1  # O(1) - Decrement the loop counter.
    print("Your data is: ", Data)  # O(1) - Print the generated data.
    return Data  # O(1) - Return the generated data list.

def ManuallyLoad():
    Data = []
    RequestLoopCount = input("Please enter the number of elements you would like to enter manually: ")  # O(1) - Input number of elements.
    while int(RequestLoopCount) > 0:
        NewItemToAddToArray = input("Please enter a number to add to the data: ")  # O(1) - Input a number.
        Data.append(int(NewItemToAddToArray))  # O(1) - Append the number to the data list.
        RequestLoopCount = int(RequestLoopCount) - 1  # O(1) - Decrement the loop counter.
    print("Your data is: ", Data)  # O(1) - Print the entered data.
    return Data  # O(1) - Return the data list.

def DataLoader(Data):
    avlTree = AVLTree()  # O(1) - Create a new AVLTree instance.
    for e in Data:
        avlTree.insert(e)  # O(log n) - Insert an element into the AVL tree.
    return avlTree  # O(1) - Return the populated AVL tree.

def CountingSort(Data):
    if not Data:
        return []  # O(1) - Return an empty list if data is empty.
    UnsortedArray = Data
    FinalArray = [0] * len(UnsortedArray)  # O(n) - Initialize the final sorted array.
    max_element = max(UnsortedArray)  # O(n) - Find the maximum element.
    min_element = min(UnsortedArray)  # O(n) - Find the minimum element.
    range_of_numbers = max_element - min_element + 1  # O(1) - Compute the range of the numbers.
    count = [0] * range_of_numbers  # O(k) - Initialize the count array.
    for num in UnsortedArray:
        count[num - min_element] += 1  # O(n) - Count each number.
    for i in range(1, len(count)):
        count[i] += count[i - 1]  # O(k) - Accumulate counts.
    for num in reversed(UnsortedArray):
        FinalArray[count[num - min_element] - 1] = num  # O(n) - Place each number in its correct position.
        count[num - min_element] -= 1  # O(1) - Decrement the count.
    return FinalArray  # O(1) - Return the sorted array.

def CreateTopDownBSTInput(sorted_input, L, R):
    if L > R:
        return None  # O(1) - Base case for recursion.
    mid = (L + R) // 2  # O(1) - Calculate the middle index.
    root = TreeNode(sorted_input[mid])  # O(1) - Create a new TreeNode with middle element.
    root.left = CreateTopDownBSTInput(sorted_input, L, mid - 1)  # Recursion - O(log n) - Construct left subtree.
    root.right = CreateTopDownBSTInput(sorted_input, mid + 1, R)  # Recursion - O(log n) - Construct right subtree.
    return root  # O(1) - Return the constructed BST root.

def GetBSTData(root):
    data = []
    if root:  # O(1) - Check if the current node is not None.
        data.append(root.element)  # O(1) - Append the current node's element to the list.
        data.extend(GetBSTData(root.left))  # O(n/2) - Recursively get data from the left subtree.
        data.extend(GetBSTData(root.right))  # O(n/2) - Recursively get data from the right subtree.
    return data  # O(1) - Return the list of data collected.

def ExitMenu():
    return False # O(1) - Returns False to indicate exit.

def SecondMenu(SecondMenuActive, Data):
    # Assuming CountingSort and AVLTree have been correctly implemented
    sorted_input = CountingSort(Data)  # O(n + k) - Sort the data using Counting Sort.
    Root = CreateTopDownBSTInput(sorted_input, 0, len(sorted_input) - 1)  # O(n) - Create a BST from the sorted data.
    avlTree = DataLoader(GetBSTData(Root))  # O(n log n) - Load the sorted data into an AVL tree.

    while SecondMenuActive:  # O(1) - Check if the second menu is active.
        os.system('clear')  # O(1) - Clear the console.
        print("Please Pick One Of The Following Options: ")
        print("    - Option 1: Display the AVL tree, showing the height and balance factor for each node.")
        print("    - Option 2: Print the pre-order, in-order, and post-order traversal sequences of the AVL tree.")
        print("    - Option 3: Print all leaf nodes of the AVL tree, and all non-leaf nodes (separately).")
        print("    - Option 4: Insert a new integer key into the AVL tree.")
        print("    - Option 5: Delete an integer key from the AVL tree.")
        print("    - Option 6: Return to the level-1 menu.")
        print("------------------------------------------")
        
        SecondMenuInput = input("Please enter a menu option number: ")  # O(1) - Get user input for menu option.

        if SecondMenuInput == "1":
            avlTree.DisplayBST()  # O(n) - Display the AVL tree visually.
            input("Press Any Key To Return To The Main Menu")  # O(1) - Pause until user input.

        elif SecondMenuInput == "2":
            print("\nPre order:")
            avlTree.preorder()  # O(n) - Print pre-order traversal.
            print("\n")
            print("\nIn order:")
            avlTree.inorder()  # O(n) - Print in-order traversal.
            print("\n")
            print("\nPost order:")
            avlTree.postorder()  # O(n) - Print post-order traversal.
            print("\n")
            input("\nPress Any Key To Return To The Main Menu")  # O(1) - Pause until user input.

        elif SecondMenuInput == "3":
            print("\nLeaf nodes in order:")
            avlTree.leaf_inorder()  # O(n) - Print all leaf nodes in-order.
            print("\n")
            print("\nNon leaf nodes in order:")
            avlTree.non_leaf_inorder()  # O(n) - Print all non-leaf nodes in-order.
            print("\n")
            input("\nPress Any Key To Return To The Main Menu")  # O(1) - Pause until user input.

        elif SecondMenuInput == "4":
            UserInput = input("Please enter a number to be added: ")  # O(1) - Get number to add.
            avlTree.insert(int(UserInput))  # O(log n) - Insert new key into AVL tree.
            avlTree.DisplayBST()  # O(n) - Display the AVL tree visually.
            input("Press Any Key To Return To The Main Menu")  # O(1) - Pause until user input.

        elif SecondMenuInput == "5":
            UserInput = input("Please enter a number to be removed: ")  # O(1) - Get number to remove.
            avlTree.deleteRec(avlTree.root, int(UserInput))  # O(log n) - Delete a key from AVL tree.
            avlTree.DisplayBST()  # O(n) - Display the AVL tree visually.
            input("Press Any Key To Return To The Main Menu")  # O(1) - Pause until user input.

        elif SecondMenuInput == "6":
            SecondMenuActive = ExitMenu()  # O(1) - Exit the second menu.

        else:
            print("\nError Not An Option") # O(1) - Return to the current menu.

def FirstMenu(Active):
    while Active:  # O(1) - Loop as long as the Active flag is true.
        os.system('clear')  # O(1) - Clear the console for clean menu display.

        Data = []  # O(1) - Initialize an empty list to store integers.

        # Print the main menu options
        print("Please Pick One Of The Following Options: ")
        print("    - Option 1 : Pre-load a sequence of integers to build an AVL tree.")
        print("    - Option 2 : Manually enter integer keys one by one to build an AVL tree.")
        print("    - Option 3 : Exit Program.")
        print("------------------------------------------")

        FirstMenuInput = input("Please enter a menu option number: ")  # O(1) - Prompt user for a menu option.

        if FirstMenuInput == "1":  # O(1) - Check if the user chose option 1.
            Data = Preload()  # O(n) - Call Preload function to get pre-generated data.
            SecondMenu(True, Data)  # O(m) - Pass the data to the SecondMenu for further operations, where m is the menu logic complexity.
            FirstMenuInput = None  # O(1) - Reset FirstMenuInput for safety.

        elif FirstMenuInput == "2":  # O(1) - Check if the user chose option 2.
            Data = ManuallyLoad()  # O(n) - Call ManuallyLoad function to get user-entered data.
            SecondMenu(True, Data)  # O(m) - Pass the data to the SecondMenu for further operations.
            FirstMenuInput = None  # O(1) - Reset FirstMenuInput for safety.

        elif FirstMenuInput == "3":  # O(1) - Check if the user chose option 3.
            Active = ExitMenu()  # O(1) - Call ExitMenu to update the Active flag based on the user's choice to exit.

        else:
            print("\nError Not An Option") # O(1) - Return to the current menu.

# Entry point of the program
if __name__ == "__main__":
    FirstMenu(True)  # O(1) - Start the FirstMenu with Active set to True.