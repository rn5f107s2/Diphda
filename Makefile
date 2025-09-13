all:
	nvcc src/main.cpp src/games/chess/position.cpp src/ucioption.cpp src/uci.cpp src/games/chess/attacks.cpp src/games/chess/movegen.cpp src/search/search.cpp src/search/node.cpp src/search/evaluator.cpp src/network/cudnn.cu -o Diphda -lcudnn -std=c++20 -O3
