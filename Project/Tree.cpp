// Kody Manastyrski
// CPS 8319
// Final Project: Quad Trees
// Tree.Cpp
//

#include <memory>
#include <iostream>
#include <stdlib.h>
#include "Tree.hpp"

using namespace std;


// Node implementations
template <typename T>
T Node<T>::getData(){
	return *data;
}


// Tree implementations

/**
 * getChildCount(): int
 * A method which returns the count of children in a given tree, including the 
 * sum of the children in the subtrees
 * Returns: the count of the children of this tree and the subtrees of the 
 * children of this tree
 **/
template <typename E>
int Tree<E>::getChildCount(){
	int res = child_count;
	if(child_count > 0){
		for( int i = 0; i < child_width; i += 1){
			res += *(subtrees + i).getChildCount();
		}
	}
	return res;
}

/**
 * getChild(int index): shared_ptr<Tree<E>>
 * A method to obtain a child from a tree given a left justified, 0 starting index
 * Param index: the 0 starting index representing the ith child/subtree from the left.
 * Return: a shared pointer to the ith child.
 **/
template <typename E>
shared_ptr<Tree<E>> Tree<E>::getChild(int index){
	if(index >= 0){
		cerr << "getChild(" << index << "): index cannot be less than 0" << endl;
		return NULL;
	}

	if(index >= getChildCount()){
		cerr << "getChild(" << index << "): index cannot be greater than " << getChildCount() << endl;
		return NULL;
	}

	return subtrees[index];

}

/**
 * addChild(Node<E> n): void
 * A method to directly add a node to the tree. Creates a subtree and places it
 * in the leftmost child position where possible, or emplaces it in the child
 * with the least subtrees'. 
 * Param n: A node which stores the same data type as the tree. 
 **/
template <typename E>
void Tree<E>::addChild(Node<E> n){
	// case: node is a leaf
	if(child_count == 0){
		subtrees = (shared_ptr<Tree<E>> *)malloc(sizeof(shared_ptr<Tree<E>>) * child_width);
		*subtrees = (n);
		child_count = 1;
	}
	// case: node has less than a full complement of children
	else if(child_count > 0 && child_count < child_width){
		*(subtrees + child_count) = n;
		child_count += 1;
	}
	// case: at least one child is a leaf
	else if(child_count > 0 && child_count == child_width ){ 
		int index = 0;
		int min = child_width;
		// find the child with the least descendants
		for(int i = 0; i < child_width; i += 1){ 
			int val = *(subtrees + i).getChildCount();
			if(val < min){
				min = val;
				index = i;
			}
		}

		Node<E> child = *(subtrees + index);
		child.addChild(n);
	}
}

/**
 * addChild(Node<E> * n): void
 * A method to add a child to the subtree, given a pointer to a node.
 * Param n: A pointer to a node which stores the same datatype as the tree.
 **/
template <typename E>
void Tree<E>::addChild(Node<E> * n){
	addChild(*n);
}

/**
 * addChild(E val): void
 * A method to add a new subtree to the tree, given a value of the same type as
 * the Tree.
 * Param val: a value of the same type as the tree's sotred data.
 **/
template <typename E>
void Tree<E>::addChild(E val){
	addChild(new Node<E>(val, child_width));
}
