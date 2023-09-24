# neural-reader
Neural network writted in C to recognize handwritten letters

## Download and Compiling

1. Clone this repository
```
git clone https://github.com/johnsoupir/neural-reader.git
```

2. Change directory
```
cd neural-reader
```

3. Compile the code
```
make all
```

If this fails because make is not installed, run:
```
sudo apt install make
```

## Training and recognizing 

To train the network run train

```
./train
```

The network can continue training from a previous parameter file by setting the "oldDogNewTricks" variable to true.


To recognize using the pre-trained parameters run recognize

```
./recognize
```