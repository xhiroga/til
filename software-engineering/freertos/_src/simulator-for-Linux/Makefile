FREERTOS_DIR = FreeRTOS
FREERTOS_REPO = https://github.com/FreeRTOS/FreeRTOS.git
DEMO_DIR = $(FREERTOS_DIR)/FreeRTOS/Demo/Posix_GCC
BUILD_DIR = $(DEMO_DIR)/build
DEMO = $(BUILD_DIR)/posix_demo

.PHONY: all clean run

all: $(FREERTOS_DIR) $(DEMO)

$(FREERTOS_DIR):
	git -C $@ pull || git clone $(FREERTOS_REPO) $@
	git -C $@ submodule update --init --recursive

$(DEMO):
	make -C $(DEMO_DIR)

run:
	$(DEMO)

clean:
	rm -rf $(BUILD_DIR)
