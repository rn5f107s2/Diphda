all:
	nvcc src/main.cpp src/games/chess/position.cpp src/uci.cpp src/games/chess/attacks.cpp src/games/chess/movegen.cpp src/network/network.cu src/search/search.cpp src/search/node.cpp src/search/evaluator.cpp -o Diphda -std=c++20 -O3
