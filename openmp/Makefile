EXEC=lu-omp
OBJ =  $(EXEC) $(EXEC)-debug  $(EXEC)-serial

MATRIX_SIZE=8000
#MATRIX_SIZE=6
MATRIX_CHECK_SIZE=100
W :=`grep processor /proc/cpuinfo | wc -l`

CHECKER=inspxe-cl -collect=ti3 -r check
VIEWER=inspxe-gui check
REPORTER_PROBLEM=inspxe-cl -report problems -result-dir=./check/check.inspxe > reports/problems.txt
REPORTER_SUMMARY=inspxe-cl -report summary -result-dir=./check/check.inspxe > reports/summary.txt
REPORTER_OBSERVATIONS=inspxe-cl -report observations -result-dir=./check/check.inspxe > reports/observations.txt
REPORTER_STATUS=inspxe-cl -report status -result-dir=./check/check.inspxe > reports/status.txt

# flags
OPT=-O2 -g
DEBUG=-O0 -g
OMP=-fopenmp

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

report: 
	mkdir -p reports; \
	$(REPORTER_PROBLEM); \
	$(REPORTER_SUMMARY); \
	$(REPORTER_OBSERVATIONS); \
	$(REPORTER_STATUS)

