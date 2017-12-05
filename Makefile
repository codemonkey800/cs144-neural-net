BIN = project
CXX = clang++
CXXFLAGS = -std=c++1z -Wall

DEST = build
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:src/%.cpp=$(DEST)/%.o)

all: $(DEST)/$(BIN)

clean:
	@rm -rfv $(DEST)

lint: $(wildcard src/*)
	$(CXX) $(CXXFLAGS) -fsyntax-only $^

$(DEST)/$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(DEST)/%.o: src/%.cpp
	@mkdir -vp $(DEST)
	$(CXX) $(CXXFLAGS) -c -o $@ $^

.PHONY: all clean lint

