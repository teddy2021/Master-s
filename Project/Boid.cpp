// Kody Manastyrsk
// CP 8319
// Computational Geometry
// Boid cpp
//

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include "NDObject.hpp"
#include <math.h>
#ifdef BOID
#include "Boid.hpp"
using namespace std;
using namespace glm;

vec2 * Boid::getForward(){
	return &forwards;
}

mat2 * Boid::getModel(){
	return &model;
}

vec2 Boid::updateMovement(NDObject<Boid> *others, int n){

	vec2 top = {0, 1};
	vec2 left = {-1, 0};
	vec2 right = {1, 0};
	vec2 acceleration = {0,0};
	vec2 selfTop = model * top;
	accum = 0;

	for(int i = 0; i < n; i += 1){
		Boid other = *(others[i].getObject());
		mat2 otherModel = *(other.getModel())
		vec2 otherTop = otherModel * top;
		vec2 otherLeft = otherModel * left;
		vec2 otherRight = otherModel * right;
		vec2 otherForwards = *(other.getForward());

		inter_count = 0;

		bool f1 = sameSide(forwards, otherRight, otherTop, otherLeft) < 0;
		bool f2 = sameSide(forwards, otherTop, otherLeft, otherRight) < 0;
		bool f3 = sameSide(forwards, otherLeft, otherRight, otherTop) < 0;
		bool f4 =  sameSide(selfTop, otherRight, otherTop, otherLeft) < 0;
		bool f5 =  sameSide(selfTop, otherTop, otherLeft, otherRight) < 0;
		bool f6 =  sameSide(selfTop, otherLeft, otherRight, otherTop) < 0;

		bool f5 =   sameSide(otherTop, otherRight, forwards, selfTop) > 0;
		bool f6 =  sameSide(otherLeft, otherRight, forwards, selfTop) > 0;

		if(( f1 && f2  ) || (f3  && f4 ) || (f5  && f6 ) || // another entity is in front of this one
				(f7 && f9 )){
			if(forwards.dot(otherForwards) < 0){ // the entity is travelling opposite this one, avoid it
				acceleration -= otherForwards;
			}
			else{
				acceleration += otherForwards; // the entity is travelling parallel this one, join with it
			}
			accum += 1
		}
		return acceleration/accum;
	}
}

bool sameSide(vec2 line, vec2 origin, vec2 point1, vec2 point2){
	// first defines a line, second defines a poitn.
	float dx = line[0] - origin[0];
	float dy = line[1] - origin[1];
	float z1 = dx * (point1[1] - origin[1]) - 
		(point1[0] - origin[0]) * dy;
	float z2 = dx * (point2[1] - origin[1]) - (point2[0] - origin[0]) * dy;
	return z1 * z2;
	
}

void Boid::setForward(vec2 direction){
	forwards = direction;
}

#endif
