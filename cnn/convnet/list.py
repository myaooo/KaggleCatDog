"""
A Doubly Linked List
"""

class ListNode:
    def __init__(self, prev=None, next=None, data=None, container=None):
        self.prev = prev
        self.next = next
        self.data = data
        self.container = container


class DoublyLinkedList:
    def __init__(self, node=None):
        self.front = node
        self.back = node
        self.size = 0
        self.end = None

    def push_back_node(self, node):
        if self.size == 0:
            self.front = self.back = node
        else:
            self.back.next = node
            node.prev = self.back
            self.back = node
        self.size += 1

    def push_front_node(self, node):
        if self.size == 0:
            self.front = self.back = node
        else:
            self.front.prev = node
            node.next = self.front
            self.front = node
        self.size += 1

    def pop_front_node(self):
        if self.size == 0:
            return None
        elif self.size == 1:
            node = self.front
            self.front = self.back = None
            self.size -= 1
            return node
        else:
            node = self.front
            self.front = node.next
            self.front.prev = None
            self.size -= 1
            return node

    def pop_back_node(self):
        if self.size == 0:
            return None
        elif self.size == 1:
            node = self.back
            self.front = self.back = None
            self.size -= 1
            return node
        else:
            node = self.back
            self.back = node.prev
            self.back.next = None
            self.size -= 1
            return node


if __name__ == '__main__':
    list = DoublyLinkedList()
    list.push_back_node(ListNode(data=1))
    list.push_front_node(ListNode(data=2))
    print(list.front.data, list.back.data)
    list.pop_back_node()