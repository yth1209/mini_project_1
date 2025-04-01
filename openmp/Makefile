EXEC=lu-omp
OBJ =  $(EXEC) $(EXEC)-debug  $(EXEC)-serial

MATRIX_SIZE=2000
#MATRIX_SIZE=6
MATRIX_CHECK_SIZE=100
W=4

CHECKER=inspxe-cl -collect=ti3 -r check

# flags
OPT=-O2 -g
DEBUG=-O0 -g
OMP=-fopenmp
PERF=-fno-omit-frame-pointer

all: $(OBJ)

# build the debug parallel version of the program
$(EXEC)-debug: $(EXEC).cpp
	g++ $(DEBUG) $(OMP) -o $(EXEC)-debug $(EXEC).cpp -lrt


# build the serial version of the program
$(EXEC)-serial: $(EXEC).cpp
	g++ $(OPT) $(NOWARN) -o $(EXEC)-serial $(EXEC).cpp -lrt -lgomp

# build the optimized parallel version of the program
$(EXEC): $(EXEC).cpp
	g++ $(OPT) $(OMP) -o $(EXEC) $(EXEC).cpp -lrt

$(EXEC)-perf: $(EXEC).cpp
	g++ $(OPT) ${PERF} $(OMP) -o $(EXEC)-perf $(EXEC).cpp -lrt

#run the optimized program in parallel
runp: $(EXEC)
	@echo use make runp W=nworkers
	./$(EXEC) $(MATRIX_SIZE) $(W)

#run the serial version of your program
runs: $(EXEC)-serial
	@echo use make runs
	./$(EXEC)-serial $(MATRIX_SIZE) 1

#run the optimized program with thread checker
check: $(EXEC)
	@echo use make check W=nworkers
	$(CHECKER) ./$(EXEC) $(MATRIX_SIZE) $(W)

#view the thread checker result
view:
	$(VIEWER)


clean:
	/bin/rm -rf $(OBJ) check reports/*

perf_stat: $(EXEC)-perf
	@echo "perf report"
	perf stat ./$(EXEC)-perf $(MATRIX_SIZE) $(W)

perf_record: $(EXEC)-perf
	@echo "perf report"
	perf record -g ./$(EXEC)-perf $(MATRIX_SIZE) $(W); \
	perf report;