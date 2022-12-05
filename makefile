include=-I../../include
libpath=-L../../lib
libs=-lNatNet -lGLU -lGL -lglut -lstdc++ -lm -lglfw -lGLEW -ldl -lpthread -lgomp #-lHL -lHLU -lHDU -lHD

all:multiRigidTracking

multiRigidTracking: 
	g++ multiRigidTracking.cpp -fopenmp $(include) $(libpath) $(libs) -o multiRigidTracking

.PHONY: clean
clean:
	@rm -f ./multiRigidTracking
