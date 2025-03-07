# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
NEURALNET_DIR = $(SRC_DIR)/neuralnet

# Source files
SRCS = $(SRC_DIR)/main.cpp \
       $(NEURALNET_DIR)/NeuralNetwork.cpp \
       $(NEURALNET_DIR)/Layer.cpp \
       $(NEURALNET_DIR)/Neuron.cpp

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Executable target
TARGET = $(BIN_DIR)/neural_net

# Default rule
all: $(TARGET)

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)  # Ensure the directory exists
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run the program
run: all
	./$(TARGET)

