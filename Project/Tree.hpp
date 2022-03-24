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
		shared_ptr<T>* children;
		int child_count;
		int child_width;
	protected:
		void setData(T val){
			*data=val;
		}

	public:
		explicit Node(T val = NULL, int children = 2){data = val; child_count = 0; child_width = children;}
		T& operator*(){return * data;}
		~Node(){ delete(data); if(child_count > 0){ delete(children); } }
		void addChild(Node n);
		void addChild(Node *n);
		void addChild(T val);
		void addChild(int index, T val);
		shared_ptr<T> getChild(int index);
		T getData();
		int getChildCount();
};


template <typename E>
class Tree{
	private:
		shared_ptr<E> root;
		int depth;
		int resolution;
	
	protected:

	public:

};
#endif
