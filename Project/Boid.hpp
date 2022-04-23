// Kody Manastyrski
// Cp 8319 
// Computational Geometry
// Boid hpp
//

#include <glm/glm.hpp>
#include <cstdlib>
#include "NDObject.hpp"
#ifndef BOID
#define BOID

class Boid{

	private:
		vec2 forwards;
		mat2 * model;


	public:
		Boid(){ 
			forwards = {
				(rand() % 100)/(rand() % 100) - (rand() % 1), 
				(rand() % 100)/(rand() % 100) - (rand() % 1)
			};
		}
		vec2 updateMovement(NDObject<Boid> *others, int n);
		mat2 * getModel(vec2 position, vec2 acceleration);
		void updateModel();
		vec2 * getForward();
		void setForward(vec2 direction);
};
#endif
