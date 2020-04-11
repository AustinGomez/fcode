CC = g++
CFLAGS = -Wall -w
SRCS = image.cpp
PROG = build/main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
