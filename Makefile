all:
	nvcc src/main.cpp src/games/chess/position.cpp src/uci.cpp src/games/chess/attacks.cpp src/games/chess/movegen.cpp src/network/network.cu -o Diphda -O3
