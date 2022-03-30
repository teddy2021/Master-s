// Kody Manastyrski
// CPS 8319
// Final Project: Quad Trees
// Tree.hpp
//


#include <memory>

#ifndef TREE_H
#define TREE_H
using namespace std;

template <typename T>
class Node{
	private:
		unique_ptr<T> data;
	protected:
		void setData(T val){
			*data=val;
		}

	public:
		explicit Node(T val = NULL){data = val;}
		T& operator*(){return * data;}
		~Node(){ delete(data); }
		T getData();
};


template <typename E>
class Tree{
	private:
		shared_ptr<Node<E>> root;
		shared_ptr<Tree<E>> subtrees;
		int depth;
		int resolution;
		int child_count;
		int child_width;
	
	protected:

	public:
		E& operator*(){return *root;}
		explicit Tree(int res = 2, int children = 2){ root = NULL; depth = 0; resolution = res; child_count = 0; child_width = children;}
		explicit Tree(E val, int res = 2, int children = 2){root = new Node<E>(val), resolution = res, child_count = 0; child_width = children;}
		~Tree(){ if(root != NULL){ delete(root); } }

		void addChild(Node<E> n);
		void addChild(Node<E> *n);
		void addChild(E val);
		void addChild(int index, E val);
		shared_ptr<Tree<E>> getChild(int index);
		int getChildCount();

};
#endif
