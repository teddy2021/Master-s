// Kody Manastyrski
// CP 8316
// Computational Geometry
// NDObject cpp
//

#ifdef NDOBJ
#include "NDObject.hpp"
#include <stdarg.h>
#include <iostream>
#include <glm/glm.hpp>

using namespace glm;
using namespace std;

void NDObject::setPosition(int count, ...){
	if(count > dim){
		cerr << "Error, cannot apply position of size " << count
			<< " to a position of " << dim << " dimensions\n";
		return;
	}
	va_list pos;
	va_start(pos, count);
	for(int i = 0; i < count; i += 1){
		position[i] = va_args(pos, float);
	}
	va_end(pos);
}

void NDObject::setVelocity(int count, ...){
	if(count > dim){
		cerr << "Error, cannot apply velocity of dimension " << count
			<< " to a velocity of " << dim << " dimensions.\n";
		return;
	}
	va_list vel;
	va_start(vel, count);
	for(int i = 0; i < count; i += 1){
		velocity[i] = va_args(vel, float);
	}
	va_end(vel);
}

void NDObject::setAcceleration(int count, ...){
	if(count > dim){
		cerr << "Error, cannot apply acceleration of dimension " << count 
			<< " to an acceleration of " << dim << " dimensions.\n";
		return;
	}
	va_list accel;
	va_start(accel, count);
	for(int i = 0; i < count; i += 1){
		acceleration[i] = va_arg(accel, float);
	}
	va_end(accel);
}

vec3 NDObject::getPosition(){
	return position;
}

vec3 NDObject::getVelocity(){
	return velocity;
}

vec3 NDOObject::getAcceleration(){
	return acceleration();
}

int NDObject::getDimension(){
	return dim;
}

#endif
