CXX = g++
CXXFLAGS = -Wall -g -std=c++11

TARGET = self_attention_cpu
SRCS = ../dataload.cc self_attention_cpu.cc
HEADERS = ../dataload.h

OBJS = $(SRCS:.cc=.o)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
