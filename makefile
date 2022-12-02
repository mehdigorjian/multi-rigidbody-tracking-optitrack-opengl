include=-I../../include
libpath=-L../../lib
libs=-lNatNet -lGLU -lGL -lglut -lstdc++ -lm -lglfw -lGLEW -ldl -lpthread #-lHL -lHLU -lHDU -lHD

all:multiRigidTracking

multiRigidTracking: 
	g++ multiRigidTracking.cpp $(include) $(libpath) $(libs) -o multiRigidTracking

.PHONY: clean
clean:
	@rm -f ./multiRigidTracking
