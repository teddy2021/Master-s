// Kody Manastyrski
// CP 86319
// Computation Geometry
// Comparison cpp
//

#include "QuadTree.hpp"
#include "NDObject.hpp"
#include "Boid.hpp"
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

using namespace std;

GLFWwindow * window;

void populateList(vector<NDObject<Boid>> & group, int count){
	for( int i = 0; i < count; i += 1 ){
		Boid b;
		group.push_back(NDObject<Boid>(b, 2));
	}
}


bool initGraphics(){

	glewExperimental = true;
	if( !glfwInit() ){
		fprintf(stderr, "Failed to init GLFW.\n");
		return false;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(200,200, "Boids Example", NULL, NULL);
	if(NULL == window){
		fprintf(stderr, "Failed to create OPENGL window. Intel GPUs are not 3.3 compatible.\n");
		return false;
	}

	glfwMakeContextCurrent(window);
	glewExperimental=true;
	if( glewInit() != GLEW_OK ){
		fprintf(stderr, "Failed to init GLEW.\n");
		return false;
	}

	return true;
}

void mainLoop(){
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	static const GLfloat verteBufferData[] = {
		-1.0f, -1.0f,  0.0f,
		 1.0f, -1.0f,  0.0f,
		 0.0f,  1.0f,  0.0f
	};

	GLuint vertBuffer;
	glGenBuffers(1, &vertBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verteBufferData), 
			verteBufferData, GL_STATIC_DRAW);



	do{
		glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}while( glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
			glfwWindowShouldClose(window) == 0);
}

int main(){
	// Initialize entities and containers
	int count = rand();
	vector<NDObject<Boid>> entities;
	populateList(entities, count);
	QuadTree<Boid> screenContainer(4, entities, 100,-100);

	// Setup graphics
	if( !initGraphics()){
		return -1;
	}



}
