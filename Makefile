all:
<<<<<<< HEAD
	nvcc src/main.cpp src/selfplay/dataformat.cpp src/selfplay/manager.cpp src/selfplay/selfplay.cpp src/games/chess/position.cpp src/ucioption.cpp src/uci.cpp src/games/chess/attacks.cpp src/games/chess/movegen.cpp src/network/network.cu src/search/search.cpp src/search/node.cpp src/search/evaluator.cpp src/network/cudnn.cu -o Diphda -std=c++20 -lcudnn -O3
=======
	nvcc src/main.cpp src/games/chess/position.cpp src/ucioption.cpp src/uci.cpp src/games/chess/attacks.cpp src/games/chess/movegen.cpp src/search/search.cpp src/search/node.cpp src/search/evaluator.cpp src/network/cudnn.cu -o Diphda -lcudnn -std=c++20 -O3
>>>>>>> 7c6cbff (Sparse inputs)
