all:
	nvcc -O3 ${CUFILES} ${DEF} -o ${EXECUTABLE} -DTEST=${TEST}
clean:
	rm -f *~ *.exe
