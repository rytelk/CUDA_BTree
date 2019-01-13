btree_gpu: gpu/*.cpp 
	echo "Not implemented"
test_stx: stx/*.h tests/stx/*.h tests/stx/*.cpp 
	g++ -w -g -Wall -Wextra -Werror -o btree_test_stx tests/stx/tpunit_main.cpp tests/stx/btree_test.cpp
test_gpu: stx/*.h tests/stx/*.h tests/stx/*.cpp 
	echo "Not implemented"
clean:
	rm -f *.o btree btree_test