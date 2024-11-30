# sneaky_poker_camera
This project uses a tiny camera attached to Raspberry Pi to signal to sneakily signal to a player the strength of their hand

Images captured from the camera are fed into a neural network deployed on the
Raspberry Pi itself, which will be trained to recognize the 52 different
possible playing cards. Once the cards have been identified, the statistical
probability that the player is holding the winning hand is calculated and
signaled to the player.

### Hardware
[Raspberry Pi 4 Model B 4GB
DDR4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
[Spy Camera for Raspberry Pi](https://www.adafruit.com/product/1937)
