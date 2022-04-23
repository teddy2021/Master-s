// Kody Manastyrski
// CP 8316
// Computational Geometry
// Quad Tree cpp
//

#ifndef quadtree
#include "QuadTree.hpp"

#include "DialatedNumber.hpp"
#include "NDObject.hpp"
#include <vector>
using namespace std;


template <class E>
void QuadTree<E>::Insert(NDObject<E> val){
	if(NULL == child1){
		if(NULL == stored){
			stored = new NDObject<E>[resolution + 1]();
			*stored = 1;
			stored[1] = val; 
		}
		else if(*stored == resolution){
			*stored = NDObject<E>(val);
			Divide();
		}
		else{
			stored[*stored] = NDObject<E>(val);
			*stored += 1;
		}
	}
	else{

	}
}

void QuadTree::DeleteChildren(){
	if(child1 != NULL){
		delete child1;
		delete child2;
		delete child3;
		delete child4;
	}
}

void QuadTree::Insert(NDObject<E> val){
	if( NULL == child1 ){
		if(stored != NULL && stored_count < resolution){
			stored[stored_count] = val;
		}
		else iif(NULL == stored){
				stored = (NDObject<E> *)malloc(sizeof(NDObject) * resolution);
				stored[0] = val;
				stored_count = 1;
		}
		else{
			vector<NDObject<E>> cvals1, cvals2, cvals3, cvals4;
			int midX = left > 0 ? left/pow(2, depth):-left/pow(2,depth);
			int midY = top > 0 ? top/pow(2, depth):-top/pow(2,depth);
			for(int i = 0; i < stored_count; i += 1){
				NDObject<E> item = stored[i];
				int position = 0; 
				position += item.getPosition()[0] > midX ? 1 : 0;
				position += item.getPosition()[1] > midY ? 2 : 0;
				switch(position){
					case 0:
						cvals4.push_back(item);
						break;
					case 1:
						cvals3.push_back(item);
						break;
					case 2:
						cvals2.push_back(item);
						break;
					case 3:
						cvals1.push_back(item);
						break;
					default:
						return;
				}
			}
		}
	}
}


void QuadTree::Divide(){
	child1 = new QuadTree();
	child2 = new QuadTree();
	child3 = new QuadTree();
	child4 = new QuadTree();
}

#endif
