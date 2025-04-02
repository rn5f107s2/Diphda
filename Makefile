all:
	clang++ src/main.cpp src/chess/position.cpp src/uci.cpp src/chess/attacks.cpp src/chess/movegen.cpp -o Diphda -O3
