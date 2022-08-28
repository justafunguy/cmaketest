target=a.out
ccflags=-Iheader
obj-y=/home/wg/worker/makefiletest/
src=$(wildcard *.cpp)
objs=$(patsubst %.cpp,%.o,$(src))
$(target):	
%.o:%.cpp
	$(CXX) -c $^ -o $@ $(ccflags)
clean:
	rm $(objs) -f
test:
	echo $(obj-y)
.PHONY:clean install test
.DEFAULT_GOAL:=$(target)
$(target):$(objs)
	$(CXX) $^ -o $@ 
install:
	cp a.out ../

