MKLROOT = /thinker/storage/org/liufanGroup/public_software/intel/compilers_and_libraries_2020.1.217/linux/mkl
GSLROOT = /thinker/storage/org/liufanGroup/public_lib/gsl-2.6
OBJSDIR = obj
OBJS = $(OBJSDIR)/simudata.o
SOURCE = simudata.cpp
OUT = simudata
CXXFLAGS = -std=c++11 \
		-fopenmp \
		-m64 -I$(MKLROOT)/include \
		-I$(GSLROOT)/include 
LDFLAGS = -L$(GSLROOT)/lib \
		 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt \
		 -lgsl -lgslcblas \
		 -lpthread -lm -ldl

all: $(OBJSDIR) $(OUT)

$(OBJSDIR):
		mkdir $(OBJSDIR)

$(OUT): $(OBJS)
		g++ $(OBJS) -o $(OUT) $(CXXFLAGS) $(LDFLAGS)

$(OBJS): $(SOURCE)
		g++ -c $(SOURCE) $(CXXFLAGS) -o $@

.PHONY: clean
clean:
		rm -rf $(OBJSDIR)/*.o