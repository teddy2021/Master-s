// Kody Manastyrski
// CPS 8319
// Final Project: Quad Trees
// Tree.Cpp
//

#include <memory>
#include <stdlib.h>
#include "Tree.hpp"

using namespace std;

int Node::getChildCount(){
	int res = child_count;
	if(child_count > 0){
		for( int i = 0; i < child_width; i += 1){
			res += *(children + i).getChildCount();
		}
	}
	return res;
}

void Node::addChild(Node n){
	// case: node is a leaf
	if(child_count == 0){
		children = (shared_ptr<T> *)malloc(sizeof(shared_ptr<T>) * child_width);
		* children = (n);
		child_count = 1;
	}
	// case: node has less than a full complement of children
	else if(child_count > 0 && child_count < child_width){
		*(children + child_count) = n;
		child_count += 1;
	}
	// case: at least one child is a leaf
	else if(child_count > 0 && child_count == child_width ){ 
		int index = 0;
		int min = child_width;
		// find the child with the least descendants
		for(int i = 0; i < child_width; i += 1){ 
			int val = *(children + i).getChildCount();
			if(val < min){
				min = val;
				index = i;
			}
		}

		Node child = *(children + index);
		child.addChild(n);
	}
}

T Node::getData(){
	return *data;
}


