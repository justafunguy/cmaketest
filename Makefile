target=a.out
src=$(wildcard *.cpp)
objs=$(patsubst %.cpp,%.o,$(src))
$(target):$(objs)
	$(CXX) $^ -o $@
%.o:%.cpp
	$(CXX) -c $^ -o $@
clean:
	rm $(objs) -f
.PHONY:clean install
.DEFAULT_GOAL:=$(target)
install:
	cp a.out ../

