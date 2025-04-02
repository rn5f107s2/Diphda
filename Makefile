all:
	clang++ src/main.cpp src/games/chess/position.cpp src/uci.cpp src/games/chess/attacks.cpp src/games/chess/movegen.cpp -o Diphda -O3
