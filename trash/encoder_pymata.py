
# import the API class


from pymata_aio.pymata3 import PyMata3


# ping callback function
def encoder_val(data):
    print(str(data[2]) + ' centimeters')

# create a PyMata instance
board = PyMata3(com_port="/dev/ttyACM1")

board.encoder_config(3,5, cb=encoder_val, hall_encoder=True)

while True:
    board.sleep(.001)
#    value = board.encoder_read(3)
#    print(value)

# board.sleep(.1)
board.shutdown()