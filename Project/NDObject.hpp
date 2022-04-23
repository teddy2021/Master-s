// Kody Manastyrski
// CP 8316
// Computational Geometry
// ND Object hpp
//

#ifndef NDOBJ
#define NDOBJ

#include <stdarg.h>
#include <glm/glm.hpp>

using namespace std;
using namespace glm;

template <typename E>
class NDObject{
	private:
		E object;
		int dim;
		vec3 position;
		vec3 velocity;
		vec3 acceleration;


	public:
		NDObject(int dimensions){ 
			position = vec3(0,0,0);
			velocity = vec3(0,0,0);
			acceleration = vec3(0,0,0);
			dim=dimensions; 
		}

		NDObject(E val, int dimensions){
			object = val;
			position = vec3(0,0,0);
			velocity = vec3(0,0,0);
			acceleration = vec3(0,0,0);
			dim = dimensions;
		}
	
		void setPosition(int count, ...);
		void setVelocity(int count, ...);
		void setAcceleration(int count, ...);

		vec3 getAcceleration();
		vec3 getVelocity();
		vec3 getPosition();
		int getDimensions();

		E *getObject(){
			return &object;
		}


};

#endif
