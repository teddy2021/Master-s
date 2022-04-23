cmake_minimum_required(VERSION 3.23)


project(Comparison)

set(OpenGL_GL_PREFERENCE GLVND)

find_package(glfw3 REQUIRED)
if(glfw3_FOUND)
	message("Found glfw3 at '${glfw_INCLUDE_DIR}'")
endif()
find_package(glm REQUIRED)
if(glm_FOUND)
	message("Found glm at ${GLM_INCLUDE_DIRS}")
endif()
find_package(GLEW REQUIRED)
if(GLEW_FOUND)
	message("Found GLEW at ${GLEW_INCLUDE_DIRS}")
endif()
find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
	message("Found OpenGL at ${OPENGL_INCLUDE_DIR}")
endif()

add_executable(Comparison
	QuadTree.cpp QuadTree.hpp
	NDObject.cpp NDObject.hpp
	Boid.cpp Boid.hpp
	Comparison.cpp
	)

target_link_libraries(Comparison
	glfw
	glm
	GLEW
	OpenGL::GL
	)
