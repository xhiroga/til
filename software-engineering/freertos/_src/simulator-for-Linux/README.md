# FreeRTOS Simulation

```shell
# macOS prerequisite
brew install libpcap
uname -s | grep -q "Darwin" && uname -m | grep -q "arm64" && arch -x86_64 zsh

make
make run
```

## References

- [Posix/Linux Simulator Demo for FreeRTOS using GCC](https://www.freertos.org/Documentation/02-Kernel/03-Supported-devices/04-Demos/03-Emulation-and-simulation/Linux/FreeRTOS-simulator-for-Linux)
- [FreeRTOSなるもの](https://note.com/oraccha/n/nbb424fb9bf77)
