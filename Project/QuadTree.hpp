// Kody Manastyrskia
// CP 8316
// Computational Geometry
// Quad Tree hpp
//

#ifndef quadtree
#define quadtree

#include "Tree.hpp"
#include "NDObject.hpp"
#include "DialatedNumber.hpp"
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>


using namespace std;

template <typename E>
class QuadTree: public Tree<E>{

	private:
		QuadTree<E> * child1; // Child for quadrant 1, sin +, cos +
		QuadTree<E> * child2; // Child for quadrant 2, sin +, cos -
		QuadTree<E> * child3; // Child for quadrant 3, sin -, cos -
		QuadTree<E> * child4; // Child for quadrant 4, sin -, cos +
		int depth;
		Dialated top;
		Dialated left;
		int resolution;
		NDObject<E> *stored;
		int stored_count;
		
		void subdivide();
	
	public: 
		QuadTree(int resolution, int t, int l):Tree<E>(4, 0){ 
			depth = 1; 
			top = (Dialated)t; 
			left = (Dialated)l; 
			stored_count = 0;
		}

		QuadTree(int res, vector<NDObject<E>> val, int t, int l, int d=1):Tree<E>(res=4, val.size()){
			depth = d;
			top = (Dialated)t;
			left = (Dialated)l;
			int middleX = l>0?l/pow(2,depth):-l/pow(2,depth);
			int middleY = t>0?pow(2,t/depth):-t/pow(2,depth);
			if(res < val.size()){
				vector<NDObject<E>> c1vals, c2vals, c3vals, c4vals;
				for(int i = 0; i < val.size(); i += 1){
					NDObject<E> item = val[i];
					if(item.getPosition()[0] < middleX){
						if(item.getPosition()[1] < middleY){
							c3vals.push_back(item);
						}
						else{
							c2vals.push_back(item);
						}
					}
					else{
						if(item.getPosition()[1] < middleY){
							c4vals.push_back(item);
						}
						else{
							c1vals.push_back(item);
						}
					}
				}

				*child1 = QuadTree(res, c1vals, t, middleY, depth+1 ); // t1, l2
				*child2 = QuadTree(res, c2vals, t, l, depth+1); // t1, l1
				*child3 = QuadTree(res, c3vals, middleX, l, depth+1); // t2, l1
				*child4 = QuadTree(res, c4vals, middleX, middleY, depth+1); // t2, l2
			}
			else{
				stored = (NDObject<E> *)malloc(sizeof(NDObject<E>) * resolution);
				for(int i = 0; i < val.size(); i += 1){
					stored[i] = val[i];
					stored_count += 1;
				}
			}
		};
		
		void DeleteChildren();
		void Insert(NDObject<E> val);

};
#endif
